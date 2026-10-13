"""Retry failed checkpoints without changing successful results or Agent policy."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time

from benchmark_catalog import ROOT_DIR, input_fingerprint, load_benchmark_items
from command_parser import CommandParser
from .runner import (
    EvaluationRunner, PREFIX_METRICS, RESULTS_DIR, TOKEN_METRICS,
    _current_runs, _execute_final_action, _matches_fingerprint, _slug, write_prefix_metrics,
)


LOGGER = logging.getLogger(__name__)


def retry_checkpoint(runner, item, model, event_id):
    """Use the original Agent loop with the same independently replayed prefix."""
    events = list(item["scenario"]["events"].items())
    index = [key for key, _ in events].index(event_id)
    prefix = [{"event_id": key, **event} for key, event in events[:index + 1]]
    instance_dir = (ROOT_DIR / item["path"]).parents[2]
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="HALO-retry-") as temporary:
        episode = Path(temporary) / item["instance_id"]
        episode.mkdir()
        shutil.copy2(instance_dir / "profile.json", episode / "profile.json")
        for folder in ("data", "app"):
            shutil.copytree(instance_dir / folder, episode / folder)
        parser = CommandParser.for_instance(episode, item["instance_id"])
        try:
            for event in prefix:
                parser.execute_command({key: event[key]
                                        for key in ("app", "function", "parameter")})

            def execute_final(command):
                with tempfile.TemporaryDirectory(prefix="HALO-retry-action-") as action_temporary:
                    action_dir = Path(action_temporary) / item["instance_id"]
                    shutil.copytree(episode, action_dir)
                    action_parser = CommandParser.for_instance(action_dir, item["instance_id"])
                    try:
                        return _execute_final_action(action_parser, command, prefix[-1]["time"])
                    finally:
                        action_parser.apps["database"].connection.close()

            agent = runner._agent_checkpoint(model, parser, item["instance"], prefix, execute_final)
        finally:
            parser.apps["database"].connection.close()
    calls = agent["model_calls"]
    return {
        "event_id": event_id,
        "agent": agent,
        "metrics": {
            **{key: sum(call["usage"][key] for call in calls) for key in TOKEN_METRICS},
            "latency_ms": sum(call["latency_ms"] for call in calls),
            "intermediate_round_count": agent["intermediate_round_count"],
        },
        "retry_wall_ms": round((time.monotonic() - started) * 1000, 2),
    }


def save_retry(path, run, index, replacement, expected_bytes):
    """Atomically save one replacement, retaining every preceding failed attempt."""
    previous = run["checkpoints"][index]
    if previous["agent"]["decision"] != "error":
        raise ValueError("Only failed checkpoints may be replaced")
    if replacement["event_id"] != previous["event_id"]:
        raise ValueError("Retry event does not match the failed checkpoint")
    now = datetime.now(timezone.utc).isoformat()
    history = deepcopy(previous.get("retry_history", []))
    history.append({
        "replaced_at": now,
        "checkpoint": {key: deepcopy(value) for key, value in previous.items()
                       if key != "retry_history"},
    })
    replacement["retry_history"] = history
    updated = deepcopy(run)
    updated["checkpoints"][index] = replacement
    updated["updated_at"] = now
    points = updated["checkpoints"]
    updated["totals"] = {key: sum(c["metrics"][key] for c in points) for key in PREFIX_METRICS}
    updated["averages_per_prefix_action"] = {
        key: value / len(points) for key, value in updated["totals"].items()
    }
    attempts = points + [entry["checkpoint"] for point in points
                         for entry in point.get("retry_history", [])]
    updated["totals_including_retry_history"] = {
        key: sum(point["metrics"][key] for point in attempts) for key in PREFIX_METRICS
    }
    data = (json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode()
    if path.read_bytes() != expected_bytes:
        raise ValueError("Result changed concurrently; refusing to overwrite it")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return updated, data


def retry_run(path, item, max_attempts=3):
    """Give each currently failed checkpoint up to max_attempts fresh episodes."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    expected_bytes = path.read_bytes()
    run = json.loads(expected_bytes)
    runner = EvaluationRunner()
    fingerprint = input_fingerprint(item)
    if (not _matches_fingerprint(run, fingerprint)
            or run.get("scenario") != item["scenario"]
            or run.get("instance") != item["instance"]
            or run.get("model_settings") != runner.client.evaluation_settings(run["model"])):
        raise ValueError("Retry requires unchanged inputs, execution code, and model settings")
    if [c["event_id"] for c in run["checkpoints"]] != list(item["scenario"]["events"]):
        raise ValueError("Retry expects a saved record for every checkpoint")
    failures = [i for i, c in enumerate(run["checkpoints"]) if c["agent"]["decision"] == "error"]
    repaired = 0
    for index in failures:
        event_id = run["checkpoints"][index]["event_id"]
        for attempt in range(1, max_attempts + 1):
            LOGGER.info("RETRY [%s/%s] %s attempt=%s/%s", run["model"],
                        item["item_id"], event_id, attempt, max_attempts)
            replacement = retry_checkpoint(runner, item, run["model"], event_id)
            if input_fingerprint(item) != fingerprint:
                raise ValueError("Benchmark inputs changed during retry; result not saved")
            run, expected_bytes = save_retry(path, run, index, replacement, expected_bytes)
            success = replacement["agent"]["decision"] == "final"
            LOGGER.info("CHECKPOINT [%s/%s] %s attempt=%s/%s status=%s latency_ms=%.0f",
                        run["model"], item["item_id"], event_id, attempt, max_attempts,
                        "repaired" if success else "still_failed",
                        replacement["metrics"]["latency_ms"])
            if success:
                repaired += 1
                break
            LOGGER.warning("RETRY_FAILED %s %s: %s", item["item_id"], event_id,
                           replacement["agent"].get("error"))
    return len(failures), repaired


def run_with_retries(runner, item, model, saved_path=None):
    """Run or resume one scenario, then automatically repair failed checkpoints."""
    if saved_path is None:
        run, path = runner.run(item, model)
    else:
        path = Path(saved_path)
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["model"] != model or run["source_path"] != item["path"]:
            raise ValueError("Saved run does not match the requested model/scenario")
    if any(point["agent"]["decision"] == "error" for point in run["checkpoints"]):
        retry_run(path, item, max_attempts=3)
        run = json.loads(path.read_text(encoding="utf-8"))
    return run, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=6,
                        help="Concurrent scenarios (default: 6)")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="New full attempts per failed checkpoint in this invocation (default: 3)")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    log = args.log_file or RESULTS_DIR / "logs" / f"{_slug(args.model)}-retry.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(log, encoding="utf-8"), logging.StreamHandler()],
                        force=True)
    items = {item["path"]: item for item in load_benchmark_items()}
    jobs = [run for run, complete in _current_runs() if run["model"] == args.model and not complete]
    count = sum(c["agent"]["decision"] == "error" for run in jobs for c in run["checkpoints"])
    LOGGER.info("START failed_checkpoints=%s scenarios=%s workers=%s max_attempts=%s",
                count, len(jobs), args.workers, args.max_attempts)
    attempted = repaired = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(retry_run,
                                  RESULTS_DIR / _slug(args.model) / f"{run['run_id']}.json",
                                  items[run["source_path"]], args.max_attempts): run for run in jobs}
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                LOGGER.info("HEARTBEAT reported=%s/%s repaired=%s active_scenarios=%s",
                            attempted, count, repaired, sum(f.running() for f in pending))
            for future in done:
                run = pending.pop(future)
                try:
                    n, fixed = future.result()
                    attempted += n
                    repaired += fixed
                except Exception:
                    errors += 1
                    LOGGER.exception("RETRY_ERROR %s", run["source_path"])
                LOGGER.info("PROGRESS reported=%s/%s repaired=%s", attempted, count, repaired)
    write_prefix_metrics()
    remaining = sum(c["agent"]["decision"] == "error" for run, _ in _current_runs()
                    if run["model"] == args.model for c in run["checkpoints"])
    LOGGER.info("FINISHED attempted=%s repaired=%s remaining=%s scenario_errors=%s",
                attempted, repaired, remaining, errors)
    raise SystemExit(1 if errors or remaining else 0)


if __name__ == "__main__":
    main()
