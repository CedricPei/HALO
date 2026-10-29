"""Run proactive Agents checkpoint by checkpoint in isolated simulated state."""

import argparse
import json
import logging
import re
import shutil
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path

from command_parser import CommandParser
from benchmark_catalog import ROOT_DIR, input_fingerprint, load_benchmark_items
from env.apps._store import required_text
from llm_client import LLMClient, LLMProtocolError, LLMTransportError

from .intensity import action_intensity
from .prompt import (
    ACTION_COMMANDS,
    INSPECT_COMMANDS,
    build_agent_prompt,
)


DEFAULT_MODELS = ("gpt-5.6-sol",)
RESULTS_DIR = ROOT_DIR / "eval_results"
PREFIX_METRICS_PATH = RESULTS_DIR / "summary.json"
LOG_PATH = ROOT_DIR / "evaluation" / "evaluation.log"
LOGGER = logging.getLogger(__name__)
AUTO_TIMESTAMP_ACTIONS = {
    ("email", "create_draft"),
    ("email", "send_email"),
    ("messaging", "create_draft"),
    ("messaging", "send_message"),
    ("wallet", "create_draft"),
    ("wallet", "pay"),
    ("wallet", "refund_transaction"),
}


class AgentExecutionError(Exception):
    """An Agent command was well formed but could not be executed."""


def _slug(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def _normalize_command(command):
    """Extract one command while tolerating harmless JSON-shape variation."""
    if not isinstance(command, dict):
        raise ValueError("command must be an object")
    allowed_fields = {"app", "function", "parameter"}
    if (
        not {"app", "function"} <= command.keys()
        or command.keys() - allowed_fields
    ):
        raise ValueError(
            "command must contain app, function, and optionally parameter"
        )
    if not all(
        isinstance(command.get(field), str) and command[field].strip()
        for field in ("app", "function")
    ):
        raise ValueError("command app and function must be non-empty strings")
    app = command["app"].strip().casefold()
    function = command["function"].strip().casefold()
    parameter = command.get("parameter")
    if parameter is not None and not isinstance(parameter, dict):
        raise ValueError("command parameter must be an object")
    return {
        "app": app,
        "function": function,
        "parameter": {} if parameter is None else parameter,
    }


def _surface_command(command, available, decision):
    command = _normalize_command(command)
    name = (command["app"], command["function"])
    if name not in available:
        raise ValueError(
            f"{command['app']}.{command['function']} is unavailable for {decision}"
        )
    if name == ("database", "execute"):
        sql = command["parameter"].get("sql", "")
        verb = (
            sql.lstrip().split(None, 1)[0].upper()
            if isinstance(sql, str) and sql.strip()
            else ""
        )
        if verb not in {"SELECT", "WITH", "PRAGMA"}:
            raise ValueError("Agent database access is read-only")
    if (
        decision == "final"
        and name in AUTO_TIMESTAMP_ACTIONS
        and "timestamp" in command["parameter"]
    ):
        raise ValueError(
            f"{command['app']}.{command['function']} supplies timestamp automatically"
        )
    return command


def _execute_final_action(parser, command, timestamp):
    execution_command = command
    if (command["app"], command["function"]) in AUTO_TIMESTAMP_ACTIONS:
        execution_command = {
            **command,
            "parameter": {**command["parameter"], "timestamp": timestamp},
        }
    try:
        if (command["app"], command["function"]) == ("assistant", "notify_user"):
            # This evaluation-only output never enters the construction parser
            # or the shared Instance baseline. The run saves the notification.
            parameters = command["parameter"]
            if not isinstance(parameters, dict) or set(parameters) != {"message"}:
                raise ValueError("assistant.notify_user requires only message")
            recipient = parser.apps["email"].owner
            if not recipient:
                raise ValueError("Instance owner is not configured")
            return {
                "recipient": recipient,
                "message": required_text(parameters["message"], "message"),
                "timestamp": timestamp,
            }
        return parser.execute_command(execution_command)
    except Exception as error:
        raise AgentExecutionError(
            f"{type(error).__name__}: {error}"
        ) from error


def _execute_inspection_commands(parser, commands):
    """Execute one validated intermediate round of Agent read commands."""
    results = []
    for command in commands:
        try:
            result = parser.execute_command(command)
        except Exception as error:
            raise AgentExecutionError(
                f"{type(error).__name__}: {error}"
            ) from error
        results.append({"command": command, "result": result})
    return results


TOKEN_METRICS = (
    "input_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
)
PREFIX_METRICS = (
    *TOKEN_METRICS,
    "intermediate_round_count",
)


def _validate_response(response):
    if (
        not isinstance(response, dict)
        or not {"decision", "content"} <= response.keys()
    ):
        raise ValueError("Agent output must contain decision and content")
    decision = response["decision"]
    if isinstance(decision, str):
        decision = decision.strip().casefold()
    content = response["content"]
    if decision == "inspect":
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list) or not content:
            raise ValueError("inspect content must be a non-empty command list")
        return {
            "decision": decision,
            "content": [
                _surface_command(item, INSPECT_COMMANDS, decision)
                for item in content
            ],
        }
    if decision != "final":
        raise ValueError("decision must be inspect or final")
    if content is None:
        return {"decision": decision, "content": None}
    if isinstance(content, list) and len(content) == 1:
        content = content[0]
    if not isinstance(content, dict):
        raise ValueError("final content must be null or exactly one command object")
    content = _normalize_command(content)
    available = (
        INSPECT_COMMANDS
        if (content["app"], content["function"]) in INSPECT_COMMANDS
        else ACTION_COMMANDS
    )
    content = _surface_command(content, available, decision)
    action_intensity(content)
    return {"decision": decision, "content": content}


def _matches_fingerprint(run, fingerprint):
    """Accept an exact fingerprint or an explicitly recorded compatible revision."""
    return (
        run.get("input_fingerprint") == fingerprint
        or fingerprint in run.get("compatible_input_fingerprints", {})
    )


def _current_runs():
    """Choose one current run per job, preferring successful completion."""
    items = {item["path"]: item for item in load_benchmark_items()}
    fingerprints = {}
    selected = {}
    client = LLMClient()
    for path in sorted(RESULTS_DIR.glob("*/*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        source = run["source_path"]
        if source not in items:
            continue
        if source not in fingerprints:
            fingerprints[source] = input_fingerprint(items[source])
        if not _matches_fingerprint(run, fingerprints[source]):
            continue
        if run.get("model_settings") != client.evaluation_settings(run["model"]):
            continue
        checkpoints = run.get("checkpoints", [])
        complete = (
            [point["event_id"] for point in checkpoints]
            == list(items[source]["scenario"]["events"])
            and all(point["agent"].get("decision") == "final" for point in checkpoints)
        )
        key = (run["model"], source)
        rank = (complete, run.get("created_at", ""), path.name)
        if key not in selected or rank > selected[key][0]:
            selected[key] = (rank, run)
    return [(run, rank[0]) for rank, run in selected.values()]


def write_prefix_metrics():
    rows = []
    for run, _complete in _current_runs():
        for prefix_length, checkpoint in enumerate(run["checkpoints"], start=1):
            if checkpoint["agent"].get("decision") == "error":
                continue
            rows.append({
                "run_id": run["run_id"],
                "model": run["model"],
                "domain_id": run["domain_id"],
                "instance_id": run["instance_id"],
                "scenario_id": run["scenario_id"],
                "trigger_type": run["trigger_type"],
                "event_id": checkpoint["event_id"],
                "prefix_length": prefix_length,
                "action_intensity": checkpoint["agent"].get("action_intensity"),
                "metrics": {
                    key: checkpoint["metrics"][key]
                    for key in PREFIX_METRICS
                },
            })
    def summarize(selected):
        totals = {
            key: sum(row["metrics"][key] for row in selected)
            for key in PREFIX_METRICS
        }
        return {
            "prefix_action_count": len(selected),
            "totals": totals,
            "averages_per_prefix_action": {
                key: value / len(selected) for key, value in totals.items()
            },
        }

    report = {
        "models": {
            model: summarize([row for row in rows if row["model"] == model])
            for model in sorted({row["model"] for row in rows})
        },
        "prefix_actions": rows,
    }
    PREFIX_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=RESULTS_DIR, suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    try:
        temporary.replace(PREFIX_METRICS_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def completed_jobs():
    """Reuse only complete, successful runs of the current benchmark inputs."""
    return {(run["model"], run["source_path"])
            for run, complete in _current_runs() if complete}


class EvaluationRunner:
    def __init__(self):
        self.client = LLMClient()

    def _agent_checkpoint(self, model, parser, instance, event_prefix, execute_final):
        model_calls, checkpoint_results = [], []
        round_index = 0
        while True:
            prompt = build_agent_prompt(
                instance=instance,
                events=event_prefix,
                tool_results=checkpoint_results,
            )
            attempt_errors = []
            inspect_complete = False
            for output_attempt in range(1, 4):
                call_record = None
                try:
                    LOGGER.info("[%s] checkpoint=%s retrieval_round=%s output_attempt=%s calling API",
                                model, event_prefix[-1]["event_id"], round_index, output_attempt)
                    call = self.client.generate_json(prompt, model=model)
                    call_record = {
                        "round": round_index,
                        "output_attempt": output_attempt,
                        "raw_text": call["raw_text"],
                        "parsed": call["parsed"],
                        "usage": call["usage"],
                        "endpoint": call["endpoint"],
                        "response_id": call["response_id"],
                    }
                    model_calls.append(call_record)
                    response = _validate_response(call["parsed"])
                    if response["decision"] == "inspect":
                        results = _execute_inspection_commands(
                            parser, response["content"]
                        )
                        checkpoint_results.extend(results)
                        round_index += 1
                        inspect_complete = True
                        break
                    intensity = action_intensity(response["content"])
                    execution_result = None
                    if response["content"] is not None and intensity != "L0":
                        execution_result = execute_final(response["content"])
                    return {
                        "decision": response["decision"],
                        "content": response["content"],
                        "action_intensity": intensity,
                        "execution_result": execution_result,
                        "intermediate_round_count": round_index,
                        "inspection_trace": checkpoint_results,
                        "model_calls": model_calls,
                    }
                except (LLMProtocolError, ValueError, AgentExecutionError) as error:
                    if isinstance(error, LLMProtocolError) and error.call:
                        model_calls.append({
                            "round": round_index,
                            "output_attempt": output_attempt,
                            **error.call,
                            "protocol_error": str(error),
                        })
                    elif call_record is not None:
                        call_record["attempt_error"] = (
                            f"{type(error).__name__}: {error}"
                        )
                    attempt_errors.append(f"{type(error).__name__}: {error}")
                    LOGGER.warning(
                        "[%s] Agent final-round attempt %s/3 failed; %s: %s",
                        model,
                        output_attempt,
                        "stopping" if output_attempt == 3 else "retrying final round",
                        error,
                    )
            if not inspect_complete:
                return {
                    "decision": "error",
                    "error": attempt_errors[-1],
                    "content": None,
                    "action_intensity": None,
                    "intermediate_round_count": round_index,
                    "inspection_trace": checkpoint_results,
                    "model_calls": model_calls,
                }

    def run(self, item, model):
        instance_dir = ROOT_DIR / "domains" / item["domain_id"] / "instances" / item["instance_id"]
        fingerprint = input_fingerprint(item)
        if json.loads((ROOT_DIR / item["path"]).read_text()) != item["scenario"]:
            raise ValueError("Scenario changed after catalog loading; reload before evaluation")
        if json.loads((instance_dir / "profile.json").read_text()) != item["instance"]:
            raise ValueError("Profile changed after catalog loading; reload before evaluation")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = _slug(f"{timestamp}-{model}-{item['instance_id']}-{item['scenario_id']}")
        checkpoints = []
        events = list(item["scenario"]["events"].items())
        if any("time" not in event for _, event in events):
            raise ValueError(
                f"{item['scenario_id']} uses the legacy Event schema without time"
            )
        for checkpoint_index, (event_id, _) in enumerate(events):
            LOGGER.info(
                "[%s/%s/%s] checkpoint %s/%s",
                model,
                item["instance_id"],
                item["scenario_id"],
                checkpoint_index + 1,
                len(events),
            )
            event_prefix = []
            with tempfile.TemporaryDirectory() as temporary:
                episode_dir = Path(temporary) / item["instance_id"]
                episode_dir.mkdir()
                shutil.copy2(instance_dir / "profile.json", episode_dir / "profile.json")
                for folder in ("data", "app"):
                    shutil.copytree(instance_dir / folder, episode_dir / folder)
                parser = CommandParser.for_instance(episode_dir, item["instance_id"])
                for prefix_id, prefix_event in events[:checkpoint_index + 1]:
                    event_prefix.append({"event_id": prefix_id, **prefix_event})
                    parser.execute_command(
                        {
                            key: prefix_event[key]
                            for key in ("app", "function", "parameter")
                        }
                    )
                simulation_timestamp = event_prefix[-1]["time"]

                def execute_final(command):
                    with tempfile.TemporaryDirectory() as action_temporary:
                        action_dir = Path(action_temporary) / item["instance_id"]
                        shutil.copytree(episode_dir, action_dir)
                        action_parser = CommandParser.for_instance(
                            action_dir, item["instance_id"]
                        )
                        try:
                            return _execute_final_action(
                                action_parser, command, simulation_timestamp
                            )
                        finally:
                            action_parser.apps["database"].connection.close()

                try:
                    agent = self._agent_checkpoint(
                        model=model,
                        parser=parser,
                        instance=item["instance"],
                        event_prefix=event_prefix,
                        execute_final=execute_final,
                    )
                finally:
                    parser.apps["database"].connection.close()
                calls = agent["model_calls"]
                metrics = {
                    **{
                        key: sum(call["usage"][key] for call in calls)
                        for key in TOKEN_METRICS
                    },
                    "intermediate_round_count": agent["intermediate_round_count"],
                }
                checkpoints.append({
                    "event_id": event_id,
                    "agent": agent,
                    "metrics": metrics,
                })
                LOGGER.info(
                    "[%s/%s] checkpoint %s/%s finished: %s level=%s calls=%s tokens=%s",
                    model, item["item_id"], checkpoint_index + 1, len(events),
                    agent["decision"], agent["action_intensity"], len(calls),
                    metrics["total_tokens"],
                )
        totals = {
            key: sum(checkpoint["metrics"][key] for checkpoint in checkpoints)
            for key in PREFIX_METRICS
        }
        result = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "model_settings": self.client.evaluation_settings(model),
            "domain_id": item["domain_id"],
            "instance_id": item["instance_id"],
            "scenario_id": item["scenario_id"],
            "trigger_type": item["trigger_type"],
            "source_path": item["path"],
            "input_fingerprint": fingerprint,
            "scenario": item["scenario"],
            "instance": item["instance"],
            "checkpoints": checkpoints,
            "totals": totals,
            "averages_per_prefix_action": {
                key: value / len(checkpoints)
                for key, value in totals.items()
            },
        }
        if input_fingerprint(item) != fingerprint:
            raise ValueError("Shared Instance state changed during evaluation; result not saved")
        model_dir = RESULTS_DIR / _slug(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        output = model_dir / f"{run_id}.json"
        temporary_output = output.with_suffix(".tmp")
        temporary_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_output.replace(output)
        return result, output


def main():
    from .retry import run_with_retries

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--instance", action="append", dest="instances")
    parser.add_argument("--scenario")
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Evaluate only source paths listed under selected_scenarios",
    )
    parser.add_argument("--workers", type=int, default=6,
                        help="Concurrent model/scenario jobs (default: 6)")
    parser.add_argument("--log-file", type=Path, default=LOG_PATH)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(args.log_file, mode="a", encoding="utf-8"),
                  logging.StreamHandler()],
        force=True,
    )
    items = load_benchmark_items()
    if args.selection_manifest:
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        selected_paths = {
            entry["path"] for entry in selection.get("selected_scenarios", [])
        }
        if not selected_paths:
            raise SystemExit("Selection manifest contains no selected Scenarios")
        items = [item for item in items if item["path"] in selected_paths]
    if args.instances:
        items = [item for item in items if item["instance_id"] in args.instances]
    if args.scenario:
        items = [item for item in items if item["scenario_id"] == args.scenario]
    if not items:
        raise SystemExit("No matching accepted Scenarios")
    runner = EvaluationRunner()
    failures = 0
    current = _current_runs()
    completed = {(run["model"], run["source_path"])
                 for run, complete in current if complete}
    incomplete = {
        (run["model"], run["source_path"]):
            RESULTS_DIR / _slug(run["model"]) / f"{run['run_id']}.json"
        for run, complete in current if not complete
    }
    jobs = [
        (model, item)
        for model in args.models or DEFAULT_MODELS
        for item in items
        if (model, item["path"]) not in completed
    ]
    if not jobs:
        write_prefix_metrics()
        print("All matching model/Scenario evaluations already exist.", flush=True)
        return
    started = time.monotonic()
    finished = succeeded = 0
    checkpoint_errors = 0
    model_counts = Counter(model for model, _ in jobs)
    LOGGER.info("START jobs=%s workers=%s models=%s already_complete=%s",
                len(jobs), args.workers, dict(model_counts),
                len(items) * len(args.models or DEFAULT_MODELS) - len(jobs))
    for model in model_counts:
        LOGGER.info("[%s] settings=%s", model, runner.client.evaluation_settings(model))
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        pending = {
            executor.submit(
                run_with_retries, runner, item, model,
                incomplete.get((model, item["path"])),
            ): (model, item)
            for model, item in jobs
        }
        stopping = False
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                LOGGER.info("HEARTBEAT finished=%s/%s succeeded=%s failed=%s active=%s queued=%s elapsed=%.0fs",
                            finished, len(jobs), succeeded, failures,
                            sum(f.running() for f in pending),
                            sum(not f.running() for f in pending), time.monotonic() - started)
            for future in done:
                model, item = pending.pop(future)
                if future.cancelled():
                    continue
                finished += 1
                try:
                    result, path = future.result()
                    errors = sum(point["agent"]["decision"] != "final"
                                 for point in result["checkpoints"])
                    checkpoint_errors += errors
                    if errors:
                        failures += 1
                        LOGGER.error("[%s] INCOMPLETE %s checkpoint_errors=%s result=%s",
                                     model, item["item_id"], errors, path.relative_to(ROOT_DIR))
                    else:
                        succeeded += 1
                        LOGGER.info("[%s] COMPLETE %s tokens=%s result=%s",
                                    model, item["item_id"], result["totals"]["total_tokens"],
                                    path.relative_to(ROOT_DIR))
                except Exception as error:
                    failures += 1
                    LOGGER.exception("[%s] FAILED %s: %s", model, item["item_id"], error)
                    if isinstance(error, LLMTransportError) and not stopping:
                        stopping = True
                        LOGGER.error("Transport failed; cancelling queued jobs and draining active jobs")
                        for future_pending in pending:
                            future_pending.cancel()
                LOGGER.info("PROGRESS finished=%s/%s (%.1f%%) succeeded=%s failed=%s elapsed=%.0fs",
                            finished, len(jobs), 100 * finished / len(jobs),
                            succeeded, failures, time.monotonic() - started)
    write_prefix_metrics()
    LOGGER.info("FINISHED status=%s completed=%s/%s failed=%s checkpoint_errors=%s elapsed=%.0fs",
                "failed" if failures else "complete", succeeded, len(jobs), failures,
                checkpoint_errors, time.monotonic() - started)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
