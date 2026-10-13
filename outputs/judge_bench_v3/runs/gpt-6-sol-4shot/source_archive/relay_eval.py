"""Durable HTTP-only, non-thinking relay evaluation with frozen demonstrations.

Every request reserves an ordinal on disk before sending. Resume consumes an
interrupted reservation as a transport failure, never grants fresh retry budgets,
and never reruns successful items. This job uses no local model or GPU.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import compare, fewshot, metrics, prompt, relay_report, runner


PHASES = ("first_pass", "recovery")


def _save(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(config: dict, output: Path) -> tuple[list[dict], list[dict], dict]:
    endpoint = urllib.parse.urlparse(config["endpoint"])
    if endpoint.scheme != "https" or endpoint.hostname != "api.tiantoken.com" or endpoint.path != "/v1/chat/completions":
        raise ValueError("Expected the authorized Tiantoken HTTPS chat endpoint")
    if config["model"] != "gpt-6-sol" or config["reasoning_effort"] != "none":
        raise ValueError("This frozen run requires exact gpt-6-sol with reasoning disabled")
    if config["concurrency"] < 1 or config["transport_retries"] != 2 or config["recovery_structure_retries"] != 2:
        raise ValueError("Invalid frozen concurrency/retry policy")
    manifest = json.loads((output / "example_manifest.json").read_text())
    frozen = json.loads((output / "frozen_prompt.json").read_text())
    if (manifest["test_sha256"] != _sha(Path(config["test"]))
            or manifest["train_sha256"] != _sha(Path(config["train"]))
            or manifest["prepared_messages_sha256"] != _sha(output / "prepared_messages.jsonl")
            or frozen["prompt_sha256"] != fewshot.PROMPT_SHA256
            or frozen["base_prompt_sha256"] != prompt.PROMPT_SHA256
            or frozen["source_sha256"] != _sha(Path(fewshot.__file__))):
        raise ValueError("Frozen data/messages/prompt contract changed")
    rows = runner._read_jsonl(Path(config["test"]))
    prepared = runner._read_jsonl(output / "prepared_messages.jsonl")
    if len(rows) != 800 or len(prepared) != 800:
        raise ValueError("Expected the complete fixed 800-item benchmark")
    if [row["item_id"] for row in rows] != [row["item_id"] for row in prepared]:
        raise ValueError("Prepared target IDs differ from test data")
    for row, item in zip(rows, prepared):
        if (item["input_context_sha256"] != runner._hash(row["input_context"])
                or item["messages_sha256"] != runner._hash(item["messages"])):
            raise ValueError("Target context/message hash mismatch")
    source_hashes = {Path(module.__file__).name: _sha(Path(module.__file__))
                     for module in (compare, fewshot, metrics, prompt, relay_report, runner)}
    source_hashes[Path(__file__).name] = _sha(Path(__file__))
    contract = {**config, "input_rows": len(rows), "input_sha256": manifest["test_sha256"],
                "train_sha256": manifest["train_sha256"],
                "prepared_messages_sha256": manifest["prepared_messages_sha256"],
                "prompt_version": fewshot.VERSION, "prompt_sha256": fewshot.PROMPT_SHA256,
                "base_prompt_sha256": prompt.PROMPT_SHA256, "schema_sha256": runner._hash(prompt.JUDGE_SCHEMA),
                "source_hashes": source_hashes, "transport": "HTTPS request/response; no WebSocket",
                "scoring_policy": "structural validity gates scoring; semantic inconsistencies are diagnostics and never trigger retries",
                "seed_strategy": runner.SEED_STRATEGY}
    return rows, prepared, contract


def _payload(item: dict, config: dict, seed: int) -> dict:
    return {"model": config["model"], "messages": item["messages"],
            "reasoning_effort": config["reasoning_effort"],
            "max_completion_tokens": config["max_completion_tokens"],
            "temperature": config["temperature"], "top_p": config["top_p"],
            "presence_penalty": config["presence_penalty"], "seed": seed,
            "response_format": prompt.RESPONSE_FORMAT, "stream": False, "store": False}


def _state_path(output: Path, item_id: str) -> Path:
    return output / "items" / (hashlib.sha256(item_id.encode()).hexdigest() + ".json")


def _result(item: dict, attempt: dict, cumulative: int) -> dict:
    fields = ("status", "prediction", "error", "finish_reason", "response_id", "usage",
              "diagnostic_consistency", "diagnostic_consistency_error", "seed", "seed_ordinal",
              "request_sha256", "started_utc", "finished_utc", "latency_seconds", "returned_model")
    return {"item_id": item["item_id"], "messages_sha256": item["messages_sha256"],
            "input_context_sha256": item["input_context_sha256"],
            "cumulative_attempts": cumulative, **{key: attempt[key] for key in fields if key in attempt}}


def _evaluate_item(item: dict, phase: str, output: Path, config: dict, key: str) -> dict:
    path = _state_path(output, item["item_id"])
    state = json.loads(path.read_text()) if path.exists() else {
        "item_id": item["item_id"], "messages_sha256": item["messages_sha256"],
        "input_context_sha256": item["input_context_sha256"], "attempts": [], "phases": {},
    }
    if any(state[field] != item[field] for field in ("item_id", "messages_sha256", "input_context_sha256")):
        raise ValueError("Saved item identity changed")
    progress = state["phases"].setdefault(phase, {"done": False})
    if progress["done"]:
        return progress["result"]
    attempts = [attempt for attempt in state["attempts"] if attempt["phase"] == phase]
    if attempts and attempts[-1]["status"] == "inflight":
        attempts[-1].update(status="transport_error", transient=True, structural=False, prediction=None,
                            error="Supervisor interrupted after reserving an HTTP attempt; delivery/result unknown. Reservation consumes transport budget.",
                            finished_utc=runner._utc())
        _save(path, state)
    while True:
        if attempts:
            last = attempts[-1]
            structural_count = sum(bool(attempt.get("structural")) for attempt in attempts)
            transient_count = sum(bool(attempt.get("transient")) for attempt in attempts)
            structure_budget = 0 if phase == "first_pass" else config["recovery_structure_retries"]
            should_retry = ((last.get("structural") and structural_count <= structure_budget)
                            or (last.get("transient") and transient_count <= config["transport_retries"]))
            if last["status"] == "ok" or not should_retry:
                result = _result(item, last, len(state["attempts"]))
                progress.update(done=True, result=result, finished_utc=runner._utc())
                _save(path, state)
                return result
            reason = "structural_response_error" if last.get("structural") else "transient_request_error"
            if last.get("transient"):
                time.sleep(min(16, 2 ** (transient_count - 1)))
        else:
            reason = "initial_request" if phase == "first_pass" else "recovery_of_previous_failure"
        ordinal = len(state["attempts"])
        seed = runner._attempt_seed(config["seed"], item["item_id"], ordinal,
                                    {attempt["seed"] for attempt in state["attempts"]})
        payload = _payload(item, config, seed)
        attempt = {"phase": phase, "attempt": ordinal + 1, "seed_ordinal": ordinal, "seed": seed,
                   "retry_reason": reason, "request_sha256": runner._hash(payload),
                   "status": "inflight", "prediction": None, "started_utc": runner._utc(), "usage": {}}
        state["attempts"].append(attempt)
        attempts.append(attempt)
        _save(path, state)  # Reserve before sending, including on process interruption.
        started = time.perf_counter()
        attempt.update(transient=False, structural=False)
        try:
            response = runner._request(config["endpoint"], key, payload=payload,
                                       timeout=config["request_timeout_seconds"])
            attempt["response"] = response
            attempt["returned_model"] = response.get("model")
            attempt["usage"] = response.get("usage") or {}
            returned = response.get("model")
            if returned and returned != config["model"] and not returned.startswith(config["model"] + "-"):
                attempt.update(status="model_mismatch", error=f"Relay returned a different model: {returned}", fatal=True)
            else:
                try:
                    attempt.update(runner._parse_response(response), status="ok")
                except (KeyError, IndexError, TypeError, ValueError) as error:
                    attempt.update(status="schema_error", structural=True, error=f"{type(error).__name__}: {error}")
                    choices = response.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        attempt["finish_reason"] = choices[0].get("finish_reason")
                    attempt["response_id"] = response.get("id")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace").replace(key, "[redacted]")
            attempt.update(status="api_error", http_status=error.code, raw_body=body,
                           error=f"HTTP {error.code}: {body[:4000]}",
                           transient=error.code in {408, 429, 500, 502, 503, 504},
                           fatal=error.code in {400, 401, 402, 403, 404})
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as error:
            attempt.update(status="transport_error", transient=True,
                           error=f"{type(error).__name__}: {error}".replace(key, "[redacted]"))
        except (json.JSONDecodeError, ValueError) as error:
            attempt.update(status="schema_error", structural=True, error=f"{type(error).__name__}: {error}")
            if isinstance(error, runner._ResponseShapeError):
                attempt["raw_body"] = error.raw_body.replace(key, "[redacted]")
        attempt.update(latency_seconds=time.perf_counter() - started, finished_utc=runner._utc())
        _save(path, state)
        if attempt.get("fatal"):
            progress.update(done=True, result=_result(item, attempt, len(state["attempts"])))
            _save(path, state)
            raise RuntimeError(attempt.get("error", "Fatal relay configuration error"))


def _run_phase(items: list[dict], phase: str, output: Path, config: dict, key: str, job: dict) -> None:
    completed = 0
    last_progress = time.monotonic()
    # A bounded batch avoids submitting all 800 billable tasks before detecting a
    # credential/model/configuration failure. Cancel tasks not yet started on error.
    executor = ThreadPoolExecutor(max_workers=config["concurrency"])
    pending = {}
    iterator = iter(items)
    def submit() -> bool:
        item = next(iterator, None)
        if item is None:
            return False
        pending[executor.submit(_evaluate_item, item, phase, output, config, key)] = item["item_id"]
        return True
    try:
        for _ in range(min(config["concurrency"], len(items))):
            submit()
        while pending:
            future = next(as_completed(pending))
            pending.pop(future)
            result = future.result()
            completed += 1
            submit()
            now = time.monotonic()
            if now - last_progress >= 15 or not pending:
                job.update(status=phase, phase_completed=completed, phase_total=len(items), checked_utc=runner._utc())
                _save(output / "job.json", job)
                print(json.dumps({"event": "progress", "phase": phase, "completed": completed,
                                  "total": len(items), "last_status": result["status"]}), flush=True)
                last_progress = now
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True)


def _export(output: Path, items: list[dict], phase: str) -> tuple[list[dict], list[dict]]:
    predictions, raw = [], []
    for item in items:
        state = json.loads(_state_path(output, item["item_id"]).read_text())
        result = state["phases"].get(phase, {}).get("result")
        if result is None and phase == "recovery":
            result = state["phases"]["first_pass"]["result"]
        if result is None:
            raise ValueError("Cannot export an unfinished phase")
        predictions.append(result)
        raw.append({"item_id": item["item_id"], "messages_sha256": item["messages_sha256"],
                    "input_context_sha256": item["input_context_sha256"], "attempts": state["attempts"]})
    return predictions, raw


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "items").mkdir(exist_ok=True)
    with (output / "job.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, prepared, contract = _validate(config, output)
        key = runner._get_api_key(config["api_key_env"], Path(config["env_file"]))
        metadata_path = output / "run_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata["config"] != contract:
                raise ValueError("Resume configuration/source differs from frozen run")
        else:
            metadata = {"config": contract, "started_utc": runner._utc(), "sessions": [], "active_wall_seconds": 0.0}
            _save(metadata_path, metadata)
        job_path = output / "job.json"
        job = json.loads(job_path.read_text()) if job_path.exists() else {
            "started_utc": runner._utc(), "completed_phases": [], "config_sha256": _sha(config_path),
        }
        job.update(supervisor_pid=os.getpid(), uid=os.getuid(), pgrp=os.getpgrp(),
                   resumed_utc=runner._utc(), uses_local_gpu=False)
        _save(job_path, job)
        if job.get("status") == "completed":
            return job
        protocol_path = output / "evaluation_protocol.json"
        protocol = {"version": "relay-four-shot-v1", "configuration_sha256": runner._hash(contract),
                    "first_pass_structure_retries": 0, "recovery_structure_retries": 2,
                    "transport_retries_per_phase": 2, "recovery_rounds": 1,
                    "retry_signal": "JSON/schema/response shape or transport/API failures; semantic labels are never retry signals",
                    "seed_strategy": runner.SEED_STRATEGY,
                    "source": "All demonstrations are training annotations; target checkpoint excluded across source models",
                    "comparison_scope": "Same new fixed 800 targets as Qwen v3; GPT uses labeled demonstrations, so this is not a matched zero-shot model comparison",
                    "thinking": "reasoning_effort=none; do not infer hidden reasoning count when relay omits token details"}
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Evaluation recovery protocol changed")
        _save(protocol_path, protocol)
        started = time.perf_counter()
        session = {"started_utc": runner._utc(), "supervisor_pid": os.getpid()}
        try:
            if "first_pass" not in job["completed_phases"]:
                _run_phase(prepared, "first_pass", output, config, key, job)
                predictions, raw = _export(output, prepared, "first_pass")
                runner._rewrite_jsonl(output / "first_pass_predictions.jsonl", predictions)
                _save(output / "first_pass_metrics.json", metrics.compute_metrics(rows, predictions, bootstrap_repeats=0))
                job["recovery_item_ids"] = [record["item_id"] for record in predictions if record["status"] != "ok"]
                job["completed_phases"].append("first_pass")
                _save(job_path, job)
            if "recovery" not in job["completed_phases"]:
                eligible = set(job["recovery_item_ids"])
                _run_phase([item for item in prepared if item["item_id"] in eligible], "recovery", output, config, key, job)
                job["completed_phases"].append("recovery")
                _save(job_path, job)
            predictions, raw = _export(output, prepared, "recovery")
            runner._rewrite_jsonl(output / "predictions.jsonl", predictions)
            runner._rewrite_jsonl(output / "raw_responses.jsonl", raw)
            session.update(finished_utc=runner._utc(), active_wall_seconds=time.perf_counter() - started, status="completed")
            metadata["sessions"].append(session)
            metadata["active_wall_seconds"] += session["active_wall_seconds"]
            performance = runner._performance(predictions, metadata["active_wall_seconds"], raw)
            performance["reasoning_tokens_note"] = "Relay omitting a separate reasoning count is recorded as unknown, not proof of zero hidden reasoning. Requests explicitly set reasoning_effort=none."
            performance["reasoning_usage_details_reported_attempts"] = sum(
                "reasoning_tokens" in (a.get("usage", {}).get("completion_tokens_details") or {})
                for r in raw for a in r["attempts"])
            result = metrics.compute_metrics(rows, predictions, bootstrap_repeats=config["bootstrap_repeats"], seed=config["seed"])
            result["performance"] = performance
            _save(output / "metrics.json", result)
            metadata.update(complete=True, finished_utc=runner._utc(), performance=performance,
                            final_status_counts=dict(Counter(record["status"] for record in predictions)),
                            first_pass_failed=len(job["recovery_item_ids"]))
            _save(metadata_path, metadata)
            relay_report.create_report(Path(config["test"]), output)
            job.update(status="completed", finished_utc=runner._utc(), completed_items=len(predictions),
                       valid_items=sum(record["status"] == "ok" for record in predictions),
                       request_attempts=performance["request_attempts"])
            _save(job_path, job)
            print(json.dumps({"event": "completed", "job": job}), flush=True)
            return job
        except BaseException as error:
            session.update(finished_utc=runner._utc(), active_wall_seconds=time.perf_counter() - started, status="failed")
            metadata["sessions"].append(session)
            metadata["active_wall_seconds"] += session["active_wall_seconds"]
            _save(metadata_path, metadata)
            job.update(status="failed", error=f"{type(error).__name__}: {error}".replace(key, "[redacted]"), finished_utc=runner._utc())
            _save(job_path, job)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
