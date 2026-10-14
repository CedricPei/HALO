"""Concurrent, resumable HTTP(S) JudgeBench inference against OpenAI-compatible APIs.

Example:
    python -m evaluation.judge_bench.runner --input validation.jsonl --output-dir run \
        --base-url http://127.0.0.1:8003/v1 --model qwen3.5-9b \
        --api-key-env QWEN35_LOCAL_API_KEY --concurrency 64 --thinking true \
        --structured-output true --max-tokens 8192
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import random
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import prompt
from .metrics import compute_metrics


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path, *, recover_partial: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if recover_partial and index == len(lines) - 1:
                break
            raise ValueError(f"Invalid JSON in {path.name}, line {index + 1}") from None
        if not isinstance(row, dict):
            raise ValueError(f"JSONL records must be objects ({path.name}, line {index + 1})")
        rows.append(row)
    return rows


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _get_api_key(name: str, env_file: Path | None) -> str:
    value = os.environ.get(name)
    if value:
        return value
    if env_file and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw = line.split("=", 1)
            if key.strip() == name:
                parts = shlex.split(raw, comments=True)
                if len(parts) != 1 or not parts[0]:
                    raise ValueError(f"Invalid or empty value for {name} in env file")
                return parts[0]
    raise ValueError(f"API key variable {name} is unavailable; set it or pass --env-file")


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return ""


class _ResponseShapeError(ValueError):
    """An HTTP response body is not an OpenAI-compatible JSON object."""

    def __init__(self, message: str, raw_body: str):
        super().__init__(message)
        self.raw_body = raw_body


def _request(
    url: str, api_key: str, *, payload: dict[str, Any] | None = None, timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=_canonical(payload) if payload is not None else None,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    try:
        result = json.loads(body)
    except json.JSONDecodeError as error:
        raise _ResponseShapeError(f"API response is not JSON: {error}", body) from error
    if not isinstance(result, dict):
        raise _ResponseShapeError("API response is not a JSON object", body)
    return result


def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("API choices must be a nonempty list with an object first choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("API choice message must be an object")
    content = _extract_text(message.get("content"))
    result = {
        "prediction": prompt.parse_prediction(content, check_consistency=False),
        "finish_reason": choice.get("finish_reason"), "response_id": response.get("id"),
        "diagnostic_consistency": True,
    }
    # Semantic label mistakes are scored unchanged, never used as retry signals.
    try:
        prompt.parse_prediction(content)
    except ValueError as error:
        result.update(diagnostic_consistency=False, diagnostic_consistency_error=str(error))
    return result


SEED_STRATEGY = (
    "Ordinal 0 uses the frozen base seed. Ordinal >=1 uses the first 16 hex digits of "
    "SHA256('judge-bench-retry-v1:' + base_seed + ':' + item_id + ':' + ordinal), "
    "modulo 2147483647; increment modulo 2147483647 on collision with any prior seed. "
    "Ordinal counts every persisted HTTP attempt for that item across invocations."
)


def _attempt_seed(base_seed: int, item_id: str, ordinal: int, used: set[int]) -> int:
    if ordinal == 0:
        return base_seed
    digest = hashlib.sha256(f"judge-bench-retry-v1:{base_seed}:{item_id}:{ordinal}".encode()).hexdigest()
    seed = int(digest[:16], 16) % 2147483647
    while seed == base_seed or seed in used:
        seed = (seed + 1) % 2147483647
    return seed


def _infer(
    row: dict[str, Any], *, args: argparse.Namespace, api_key: str,
    prior_attempts: int = 0, prior_seeds: set[int] | None = None, invocation_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Only the four approved context fields reach the model. Neither provenance nor gold do.
    context = row["input_context"]
    messages = prompt.build_messages(context)
    payload: dict[str, Any] = {
        "model": args.model, "messages": messages, "temperature": args.temperature,
        "top_p": args.top_p, "max_tokens": args.max_tokens, "stream": False,
        "presence_penalty": args.presence_penalty, "seed": args.seed,
    }
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.thinking != "auto":
        payload["chat_template_kwargs"] = {"enable_thinking": args.thinking == "true"}
    if args.structured_output:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "judge_bench", "strict": True, "schema": prompt.JUDGE_SCHEMA},
        }
    record_latency = getattr(args, "record_latency", True)
    started = time.perf_counter() if record_latency else None
    record: dict[str, Any] = {
        "item_id": row["item_id"], "status": "transport_error", "prediction": None,
        "messages_sha256": _hash(messages),
        "input_context_sha256": _hash(context), "started_utc": _utc(), "attempts": 0,
        "usage": {}, "finish_reason": None, "prior_attempts": prior_attempts,
        "invocation_id": invocation_id,
    }
    raw: dict[str, Any] = {
        "item_id": row["item_id"], "messages_sha256": record["messages_sha256"],
        "input_context_sha256": record["input_context_sha256"], "invocation_id": invocation_id,
        "prior_attempts": prior_attempts, "attempts": [],
    }
    rng = random.Random(f"{args.seed}:{row['item_id']}")
    used_seeds = set(prior_seeds or ())
    structure_retries = transport_retries = 0
    max_structure_retries = getattr(args, "max_structure_retries", 2)
    retry_reason = "recovery_of_previous_failure" if prior_attempts else "initial_request"
    while True:
        attempt = len(raw["attempts"])
        ordinal = prior_attempts + attempt
        seed = _attempt_seed(args.seed, row["item_id"], ordinal, used_seeds)
        used_seeds.add(seed)
        payload["seed"] = seed
        request_sha = _hash(payload)
        raw.setdefault("request_sha256", request_sha)
        record.update(attempts=attempt + 1, seed=seed, seed_ordinal=ordinal, request_sha256=request_sha)
        for field in ("error", "diagnostic_consistency", "diagnostic_consistency_error", "response_id"):
            record.pop(field, None)
        record.update(prediction=None, usage={}, finish_reason=None)
        attempt_started = time.perf_counter() if record_latency else None
        attempt_raw: dict[str, Any] = {
            "attempt": attempt + 1, "seed_ordinal": ordinal, "seed": seed,
            "request_sha256": request_sha, "started_utc": _utc(), "retry_reason": retry_reason,
            "usage": {},
        }
        transient = structural = False
        try:
            response = _request(
                args.base_url.rstrip("/") + "/chat/completions", api_key,
                payload=payload, timeout=args.request_timeout,
            )
            attempt_raw["response"] = response
            usage = response.get("usage")
            record["usage"] = usage if isinstance(usage, dict) else {}
            attempt_raw["usage"] = record["usage"]
            try:
                record.update(_parse_response(response), status="ok")
            except (KeyError, IndexError, TypeError, ValueError) as error:
                record.update(status="schema_error", error=f"{type(error).__name__}: {error}")
                structural = True
                # Even a truncated output retains its API finish reason and usage.
                choices = response.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    record["finish_reason"] = choices[0].get("finish_reason")
                record["response_id"] = response.get("id")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            record.update(status="api_error", error=f"HTTP {error.code}: {detail[:4000]}")
            attempt_raw.update(http_status=error.code, raw_body=detail)
            transient = error.code in {408, 429, 500, 502, 503, 504}
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as error:
            record.update(status="transport_error", error=f"{type(error).__name__}: {error}")
            attempt_raw["error"] = record["error"]
            transient = True
        except (json.JSONDecodeError, ValueError) as error:
            record.update(status="schema_error", error=f"{type(error).__name__}: {error}")
            if isinstance(error, _ResponseShapeError):
                attempt_raw["raw_body"] = error.raw_body
            structural = True
        attempt_raw["status"] = record["status"]
        for field in ("error", "finish_reason", "response_id", "diagnostic_consistency", "diagnostic_consistency_error"):
            if field in record:
                attempt_raw[field] = record[field]
        if record_latency:
            attempt_raw["latency_seconds"] = time.perf_counter() - attempt_started
        attempt_raw["finished_utc"] = _utc()
        raw["attempts"].append(attempt_raw)
        if structural and structure_retries < max_structure_retries:
            structure_retries += 1
            retry_reason = "structural_response_error"
            continue
        if transient and transport_retries < args.max_retries:
            transport_retries += 1
            retry_reason = "transient_request_error"
            time.sleep(min(16.0, 2 ** (transport_retries - 1)) + rng.random())
            continue
        break
    if record_latency:
        record["latency_seconds"] = time.perf_counter() - started
    record["finished_utc"] = _utc()
    record.update(cumulative_attempts=prior_attempts + record["attempts"],
                  structure_retries=structure_retries, transport_retries=transport_retries)
    if record_latency:
        raw["latency_seconds"] = record["latency_seconds"]
    return record, raw


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    at = (len(ordered) - 1) * quantile
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values), "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
        "p50": _percentile(values, 0.5), "p95": _percentile(values, 0.95),
    }


class GPUMonitor:
    """Keep transient samples in memory; retain only the run-level summary."""

    def __init__(self, index: int, interval: float, enabled: bool):
        self.index, self.interval, self.enabled = index, interval, enabled
        self.samples: list[tuple[float, ...]] = []
        self.errors: Counter[str] = Counter()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def start(self) -> None:
        if self.enabled:
            self.thread.start()

    def _poll(self) -> None:
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", f"--id={self.index}",
                     "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=10,
                )
                parts = result.stdout.strip().split(",")
                values = tuple(float(part.strip()) if part.strip() not in {"[N/A]", "N/A"} else float("nan") for part in parts)
                if len(values) != 5:
                    raise ValueError("Unexpected nvidia-smi field count")
                self.samples.append(values)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.errors[type(error).__name__] += 1
                if isinstance(error, FileNotFoundError):
                    return
            self.stop_event.wait(self.interval)

    def finish(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.enabled:
            self.thread.join(timeout=12)
        fields = ("gpu_utilization_percent", "memory_utilization_percent", "memory_used_mib", "memory_total_mib", "power_watts")
        result: dict[str, Any] = {
            "enabled": self.enabled, "gpu_index": self.index, "interval_seconds": self.interval,
            "sample_count": len(self.samples), "poll_errors": dict(self.errors),
        }
        for i, field in enumerate(fields):
            result[field] = _distribution([sample[i] for sample in self.samples if math.isfinite(sample[i])])
        utilizations = [sample[0] for sample in self.samples if math.isfinite(sample[0])]
        result["fraction_samples_gpu_utilization_at_least_90"] = (
            sum(value >= 90 for value in utilizations) / len(utilizations) if utilizations else None
        )
        return result


def _performance(
    predictions: list[dict[str, Any]], duration: float, raw_records: list[dict[str, Any]]
) -> dict[str, Any]:
    attempts = [attempt for record in raw_records for attempt in record.get("attempts", [])]
    prompt_tokens = completion_tokens = reasoning_tokens = total_tokens = 0
    finish_reasons: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    usage_available = 0
    for attempt in attempts:
        response = attempt.get("response")
        usage = attempt.get("usage") or (response.get("usage") if isinstance(response, dict) else {}) or {}
        if not isinstance(usage, dict):
            usage = {}
        usage_available += bool(usage)
        prompt_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        completion_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        total_tokens += usage.get("total_tokens", 0) or 0
        detail = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
        if not isinstance(detail, dict):
            detail = {}
        reasoning_tokens += detail.get("reasoning_tokens", 0) or 0
        finish_reason = attempt.get("finish_reason")
        if finish_reason is None and isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                finish_reason = choices[0].get("finish_reason")
        finish_reasons[str(finish_reason)] += 1
        status = attempt.get("status")
        if status is None:
            if isinstance(response, dict):
                try:
                    _parse_response(response)
                    status = "ok"
                except (KeyError, IndexError, TypeError, ValueError):
                    status = "schema_error"
            else:
                status = "api_error" if "http_status" in attempt else "transport_error"
        statuses[status] += 1
    request_latencies = _distribution([a["latency_seconds"] for a in attempts if "latency_seconds" in a])
    return {
        "accounting": "Usage, request counts and request latency include every persisted raw HTTP attempt across all sessions, including discarded and failed attempts.",
        "active_wall_seconds": duration, "completed_requests": len(attempts),
        "request_attempts": len(attempts), "items_with_final_predictions": len(predictions),
        "requests_per_second": len(attempts) / duration if duration else None,
        "items_per_second": len(predictions) / duration if duration else None,
        "completion_tokens_per_second": completion_tokens / duration if duration else None,
        "total_tokens_per_second": total_tokens / duration if duration else None,
        "tokens": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "reasoning_tokens": reasoning_tokens, "total_tokens": total_tokens},
        "latency_seconds": request_latencies, "request_latency_seconds": request_latencies,
        "final_item_latency_seconds": _distribution([p["latency_seconds"] for p in predictions if "latency_seconds" in p]),
        "finish_reason_counts": dict(finish_reasons), "attempt_status_counts": dict(statuses),
        "final_finish_reason_counts": dict(Counter(str(p.get("finish_reason")) for p in predictions)),
        "attempts_with_reported_usage": usage_available,
        "attempts_without_reported_usage": len(attempts) - usage_available,
        "reasoning_tokens_note": "Reasoning tokens are zero if the serving API does not report a separate count.",
    }


def _boolean(value: str) -> bool:
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--server-config", type=Path, help="JSON object containing served settings and model artifact hashes")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--record-latency", type=_boolean, default=True, help="Disable per-request/item latency measurement and output when false")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, help="Model-specific sampling parameter; omitted unless explicitly set")
    parser.add_argument("--thinking", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--structured-output", type=_boolean, nargs="?", const=True, default=False)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-structure-retries", type=int, default=2,
                        help="Additional retries after malformed JSON/schema/API response shape; never retry semantic label inconsistencies")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=1.0)
    parser.add_argument("--gpu-monitor", type=_boolean, nargs="?", const=True, default=True)
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly rerun failed IDs without changing the frozen prompt")
    parser.add_argument("--progress-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if urllib.parse.urlparse(args.base_url).scheme not in {"http", "https"}:
        parser.error("Only HTTP(S) APIs are permitted")
    if args.concurrency < 1 or args.max_tokens < 1 or args.max_retries < 0 or args.max_structure_retries < 0 or args.bootstrap_repeats < 0:
        parser.error("Concurrency/max-tokens must be positive and retries/bootstrap-repeats nonnegative")
    if args.request_timeout <= 0 or args.gpu_poll_seconds <= 0 or args.progress_seconds <= 0:
        parser.error("Timeout and polling intervals must be positive")
    if not -2 <= args.presence_penalty <= 2 or (args.top_k is not None and args.top_k < -1):
        parser.error("Presence penalty must be in [-2,2]; top-k must be at least -1")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_bytes = args.input.read_bytes()
    rows = _read_jsonl(args.input)
    if len(rows) != 400 or any(row.get("split") != "validation" for row in rows):
        raise ValueError("Requires the frozen 400-item validation split")
    item_ids = [row["item_id"] for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Benchmark input has duplicate item IDs")
    for row in rows:
        if set(row["input_context"]) != {"background", "checkpoint_prefix", "retrieval_rounds", "candidate_action"}:
            raise ValueError(f"Unexpected context fields for {row['item_id']}")
    api_key = _get_api_key(args.api_key_env, args.env_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "run_metadata.json"
    predictions_path = args.output_dir / "predictions.jsonl"
    raw_path = args.output_dir / "raw_responses.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    prompt_source_hash = hashlib.sha256(Path(prompt.__file__).read_bytes()).hexdigest()
    schema_sha = _hash(prompt.JUDGE_SCHEMA)
    config = {
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(), "input_rows": len(rows),
        "prompt_version": prompt.PROMPT_VERSION, "prompt_sha256": prompt.PROMPT_SHA256,
        "prompt_source_sha256": prompt_source_hash, "schema_sha256": schema_sha,
        "base_url": args.base_url.rstrip("/"), "model": args.model,
        "concurrency": args.concurrency, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "top_p": args.top_p, "thinking": args.thinking,
        "presence_penalty": args.presence_penalty, "top_k": args.top_k,
        "structured_output": args.structured_output, "seed": args.seed,
        "request_timeout_seconds": args.request_timeout, "max_retries": args.max_retries,
        "scoring_policy": "structural validity gates scoring; cross-field diagnostic consistency reported separately without repair",
        "transport": "HTTP(S) request/response; no WebSocket",
    }
    if not getattr(args, "record_latency", True):
        config["record_latency"] = False
    resuming = metadata_path.exists()
    if resuming:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["config"] != config:
            differing = [key for key in config if config[key] != metadata["config"].get(key)]
            raise ValueError("Resume configuration differs: " + ", ".join(differing) + "; use a new output directory")
    else:
        if predictions_path.exists() or raw_path.exists():
            raise ValueError("Existing results have no run metadata; use a new output directory")
        metadata = {"started_utc": _utc(), "config": config, "sessions": [], "active_wall_seconds": 0.0}
        if args.server_config:
            metadata["served_configuration"] = json.loads(args.server_config.read_text(encoding="utf-8"))
        try:
            metadata["served_models"] = _request(args.base_url.rstrip("/") + "/models", api_key, timeout=min(args.request_timeout, 30))
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
            metadata["served_models_error"] = f"{type(error).__name__}: {error}"
        _atomic_json(metadata_path, metadata)
        _atomic_json(args.output_dir / "frozen_prompt.json", {
            "prompt_version": prompt.PROMPT_VERSION, "prompt_sha256": prompt.PROMPT_SHA256,
            "prompt_source_sha256": prompt_source_hash, "schema_sha256": schema_sha,
            "system_messages": [m for m in prompt.build_messages(rows[0]["input_context"]) if m["role"] == "system"],
            "schema": prompt.JUDGE_SCHEMA,
        })
    previous = _read_jsonl(predictions_path, recover_partial=True)
    by_id = {record["item_id"]: record for record in previous}
    if set(by_id) - set(item_ids):
        raise ValueError("Saved predictions include IDs outside the frozen input")
    if len(by_id) != len(previous):
        raise ValueError("Saved predictions have duplicate item IDs")
    preserved_successful_items = sum(record.get("status") == "ok" for record in by_id.values())
    if args.retry_failed:
        by_id = {key: record for key, record in by_id.items() if record.get("status") == "ok"}
    # Rewrite only on resume recovery / an explicit failed-ID retry; ensures one final record per ID.
    if predictions_path.exists():
        _rewrite_jsonl(predictions_path, list(by_id.values()))
    raw_history = _read_jsonl(raw_path, recover_partial=True)
    if raw_path.exists():
        # Preserve all prior attempts, including explicit failed-ID retries, while
        # removing an incomplete trailing write before appending new evidence.
        _rewrite_jsonl(raw_path, raw_history)
    prior_attempts_by_item: Counter[str] = Counter()
    prior_seeds_by_item: dict[str, set[int]] = {}
    for raw_record in raw_history:
        item_id = raw_record["item_id"]
        if item_id not in item_ids:
            raise ValueError("Saved raw attempts include IDs outside the frozen input")
        prior_attempts_by_item[item_id] += len(raw_record.get("attempts", []))
        for attempt in raw_record.get("attempts", []):
            # Before this retry policy, every attempt used the frozen base seed.
            prior_seeds_by_item.setdefault(item_id, set()).add(attempt.get("seed", args.seed))
    remaining = [row for row in rows if row["item_id"] not in by_id]
    session_started = _utc()
    invocation_id = _hash({"started_utc": session_started, "output_dir": str(args.output_dir),
                          "prior_invocations": len(metadata.get("retry_invocations", []))})[:20]
    invocation = {
        "invocation_id": invocation_id, "mode": "retry_failed" if args.retry_failed else "resume" if resuming else "fresh",
        "started_utc": session_started, "item_ids": sorted(row["item_id"] for row in remaining),
        "target_count": len(remaining), "preserved_successful_items": preserved_successful_items,
        "max_structure_retries": getattr(args, "max_structure_retries", 2),
        "max_transport_retries": args.max_retries, "seed_strategy": SEED_STRATEGY,
        "prior_attempts_by_item": {row["item_id"]: prior_attempts_by_item[row["item_id"]] for row in remaining},
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "retry_signal": "Only JSON/schema/API response shape and transient HTTP/transport errors; no gold labels or semantic inconsistency checks.",
        "frozen_config_unchanged": True,
        "legacy_attempt_seed": "Attempts without an explicit seed in raw records used the frozen config seed.",
    }
    # Recovery policy is append-only invocation evidence, never a retroactive edit
    # to an earlier run's frozen model/prompt/decoding configuration.
    metadata.setdefault("retry_invocations", []).append(invocation)
    _atomic_json(metadata_path, metadata)
    monitor = GPUMonitor(args.gpu_index, args.gpu_poll_seconds, args.gpu_monitor and bool(remaining))
    clock_started = time.perf_counter()
    monitor.start()
    last_progress = clock_started
    completed_this_session = 0
    interrupted = False
    try:
        with predictions_path.open("a", encoding="utf-8") as pred_handle, raw_path.open("a", encoding="utf-8") as raw_handle:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                iterator = iter(remaining)
                pending = {}
                for _ in range(min(args.concurrency, len(remaining))):
                    row = next(iterator)
                    pending[executor.submit(
                        _infer, row, args=args, api_key=api_key,
                        prior_attempts=prior_attempts_by_item[row["item_id"]],
                        prior_seeds=prior_seeds_by_item.get(row["item_id"], set()), invocation_id=invocation_id,
                    )] = row["item_id"]
                while pending:
                    done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        record, raw = future.result()
                        raw_handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
                        raw_handle.flush()
                        raw_history.append(raw)
                        pred_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        pred_handle.flush()
                        by_id[record["item_id"]] = record
                        completed_this_session += 1
                        row = next(iterator, None)
                        if row is not None:
                            pending[executor.submit(
                                _infer, row, args=args, api_key=api_key,
                                prior_attempts=prior_attempts_by_item[row["item_id"]],
                                prior_seeds=prior_seeds_by_item.get(row["item_id"], set()), invocation_id=invocation_id,
                            )] = row["item_id"]
                    now = time.perf_counter()
                    if now - last_progress >= args.progress_seconds or not pending:
                        counts = Counter(p.get("status") for p in by_id.values())
                        print(json.dumps({"event": "progress", "completed": len(by_id), "total": len(rows),
                                          "active_requests": len(pending), "statuses": dict(counts),
                                          "session_wall_seconds": round(now - clock_started, 2)}, ensure_ascii=False), flush=True)
                        last_progress = now
    except KeyboardInterrupt:
        interrupted = True
    finally:
        duration = time.perf_counter() - clock_started if remaining else 0.0
        gpu_summary = monitor.finish()
        session = {"started_utc": session_started, "finished_utc": _utc(), "wall_seconds": duration,
                   "completed_requests": completed_this_session, "gpu": gpu_summary, "interrupted": interrupted,
                   "invocation_id": invocation_id}
        invocation_raw = [raw for raw in raw_history if raw.get("invocation_id") == invocation_id]
        invocation.update(finished_utc=_utc(), completed_items=completed_this_session,
                          request_attempts=sum(len(raw.get("attempts", [])) for raw in invocation_raw),
                          interrupted=interrupted)
        session["request_attempts"] = invocation["request_attempts"]
        if remaining:
            metadata["sessions"].append(session)
            metadata["active_wall_seconds"] += duration
        predictions = [by_id[item_id] for item_id in item_ids if item_id in by_id]
        metadata["finished_utc"] = _utc()
        metadata["completed_predictions"] = len(predictions)
        metadata["complete"] = len(predictions) == len(rows)
        metadata["performance"] = _performance(predictions, metadata["active_wall_seconds"], raw_history)
        if not getattr(args, "record_latency", True):
            for name in list(metadata["performance"]):
                if "latency" in name or name.endswith("per_second"):
                    metadata["performance"].pop(name)
            metadata["performance"]["accounting"] = "Usage and request counts cover every persisted HTTP attempt; per-request/item latency is neither measured nor recorded."
        _atomic_json(metadata_path, metadata)
        report = compute_metrics(rows, predictions, bootstrap_repeats=args.bootstrap_repeats, seed=args.seed)
        report["performance"] = metadata["performance"]
        report["gpu_sessions"] = [s["gpu"] for s in metadata["sessions"]]
        report["run_complete"] = metadata["complete"]
        report["input_sha256"] = config["input_sha256"]
        report["prompt_sha256"] = config["prompt_sha256"]
        _atomic_json(metrics_path, report)
    if interrupted:
        raise KeyboardInterrupt
    print(json.dumps({"event": "finished", "total": len(rows), "valid_predictions": report["valid_predictions"],
                      "failed_predictions": report["failed_predictions"], "verdict_accuracy": report["verdict"]["accuracy"],
                      "verdict_macro_f1": report["verdict"]["macro_f1"], "metrics_path": str(metrics_path)}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    run(_arguments())


if __name__ == "__main__":
    main()
