"""Evaluate a completed adapter using the preserved baseline's frozen protocol.

This command never trains, deploys, or merges weights. An explicit thinking
override creates a separately recorded mode ablation; other decoding settings
remain frozen to the preserved baseline.
It requires a separately verified serving manifest and uses HTTP(S) through the
existing runner. A first pass is followed by the baseline's structural recovery
policy; internal label inconsistencies never cause retries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any

from . import prompt


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha() -> str:
    canonical = json.dumps(prompt.JUDGE_SCHEMA, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def validate_protocol(
    baseline: dict[str, Any], training: dict[str, Any], server: dict[str, Any],
    *, input_path: Path, adapter_path: Path,
) -> dict[str, Any]:
    if not (training.get("status") == "complete" and training.get("full_epoch_training_complete") is True
            and training.get("ready_for_judgebench_evaluation") is True):
        raise ValueError("The full training epoch and final adapter must be complete before evaluation")
    if training["config"].get("test_set_used") is not False:
        raise ValueError("Training metadata must certify no JudgeBench test use")
    frozen = baseline["config"]
    if training["config"].get("supervision_protocol") == "thinking-weighted-v2":
        if (training["config"].get("chat_template_enable_thinking") is not True
                or frozen.get("thinking") != "true" or training["config"].get("train_all") is not True
                or training.get("data", {}).get("fit", {}).get("items") != 7200
                or training.get("data", {}).get("validation", {}).get("items") != 0):
            raise ValueError("Version-2 training/evaluation thinking or all-7200 contract differs")
    if training["config"].get("supervision_protocol") == "nonthinking-weighted-v3":
        if (training["config"].get("chat_template_enable_thinking") is not False
                or frozen.get("thinking") != "false" or training["config"].get("train_all") is not True
                or training.get("data", {}).get("fit", {}).get("items") != 7200
                or training.get("data", {}).get("validation", {}).get("items") != 0
                or training["config"].get("training_scenarios") != 400):
            raise ValueError("Version-3 all-400-scenario non-thinking training/evaluation contract differs")
    expected = {
        "input_sha256": _sha(input_path), "input_rows": 800,
        "prompt_sha256": prompt.PROMPT_SHA256,
        "prompt_version": prompt.PROMPT_VERSION,
        "prompt_source_sha256": _sha(Path(prompt.__file__)),
        "schema_sha256": _schema_sha(),
    }
    for name, value in expected.items():
        if frozen.get(name) != value:
            raise ValueError(f"Frozen baseline mismatch: {name}")
    if training["config"]["prompt_sha256"] != expected["prompt_sha256"]:
        raise ValueError("Training used a different judge prompt")
    if not baseline.get("complete"):
        raise ValueError("The preserved baseline is incomplete")
    old_server = baseline["served_configuration"]
    if (training["config"].get("model_path") != old_server.get("model_path")
            or training.get("base_model_artifacts", {}).get("model_artifacts") != old_server.get("model_artifacts")):
        raise ValueError("Training base checkpoint/artifacts differ from the preserved baseline")
    identical_fields = (
        "runtime_versions", "gpu_index", "gpu_uuid", "dtype", "tensor_parallel_size",
        "max_model_len", "max_num_seqs", "max_num_batched_tokens",
        "gpu_memory_utilization", "language_model_only", "speculative_config",
    )
    for name in identical_fields:
        if name not in old_server or server.get(name) != old_server[name]:
            raise ValueError(f"Serving configuration differs from baseline: {name}")
    adapter_file = adapter_path / "adapter_model.safetensors"
    adapter_sha = _sha(adapter_file)
    adapter_config_sha = _sha(adapter_path / "adapter_config.json")
    adaptation = server.get("adaptation", {})
    if (adaptation.get("adapter_sha256") != adapter_sha
            or adaptation.get("adapter_config_sha256") != adapter_config_sha):
        raise ValueError("Serving manifest is not bound to the completed adapter weights/config hashes")
    method = adaptation.get("method")
    if method == "native_lora":
        if adaptation.get("target_coverage_verified") is not True:
            raise ValueError("Native LoRA must verify coverage of every trained target")
        if server.get("model_path") != old_server.get("model_path"):
            raise ValueError("Native LoRA must use the original base checkpoint")
        if server.get("model_artifacts") != old_server.get("model_artifacts"):
            raise ValueError("Native LoRA base artifacts differ from the frozen baseline")
    elif method == "merged_safetensors":
        if not (adaptation.get("all_base_keys_preserved") is True and adaptation.get("mtp_tensors_unchanged") is True):
            raise ValueError("Merged weights must preserve every base key and original MTP tensors")
        for name, evidence in old_server["model_artifacts"].items():
            if name.endswith(".safetensors") or name == "model.safetensors.index.json":
                continue
            if server.get("model_artifacts", {}).get(name) != evidence:
                raise ValueError(f"Merged model changed original configuration/tokenizer: {name}")
    else:
        raise ValueError("Serving adaptation.method must be native_lora or merged_safetensors")
    return {"frozen": expected, "adapter_sha256": adapter_sha,
            "adapter_config_sha256": adapter_config_sha, "adaptation_method": method,
            "training_chat_enable_thinking": training["config"].get("chat_template_enable_thinking"),
            "evaluation_thinking": frozen["thinking"], "serving_fields_matched": list(identical_fields)}


def runner_command(args: argparse.Namespace, frozen: dict[str, Any], *, recovery: bool) -> list[str]:
    command = [sys.executable, "-B", "-m", "evaluation.judge_bench.runner",
               "--input", str(args.input), "--output-dir", str(args.output_dir),
               "--model", args.model, "--base-url", frozen["base_url"],
               "--api-key-env", args.api_key_env, "--env-file", str(args.env_file),
               "--server-config", str(args.server_config),
               "--bootstrap-repeats", str(args.bootstrap_repeats),
               "--gpu-index", str(args.gpu_index),
               "--max-structure-retries", "2" if recovery else "0"]
    options = {
        "concurrency": "concurrency", "max_tokens": "max-tokens", "temperature": "temperature",
        "top_p": "top-p", "presence_penalty": "presence-penalty", "thinking": "thinking",
        "seed": "seed", "request_timeout_seconds": "request-timeout", "max_retries": "max-retries",
    }
    for field, flag in options.items():
        command.extend(["--" + flag, str(frozen[field])])
    command.extend(["--structured-output", str(frozen["structured_output"]).lower()])
    if frozen.get("top_k") is not None:
        command.extend(["--top-k", str(frozen["top_k"])])
    if recovery:
        command.append("--retry-failed")
    return command


def _write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def repair_orphan_responses(output_dir: Path, allowed_invocation_ids: set[str] | None = None) -> int:
    """Recover the final prediction of a fully persisted raw item, without new requests."""
    from .runner import _parse_response, _read_jsonl, _rewrite_jsonl

    predictions_path = output_dir / "predictions.jsonl"
    predictions = {row["item_id"]: row for row in _read_jsonl(predictions_path, recover_partial=True)}
    recovered = 0
    for raw in _read_jsonl(output_dir / "raw_responses.jsonl", recover_partial=True):
        if allowed_invocation_ids is not None and raw.get("invocation_id") not in allowed_invocation_ids:
            continue
        attempts = raw.get("attempts", [])
        if not attempts:
            continue
        last = attempts[-1]
        prior = predictions.get(raw["item_id"])
        if prior and prior.get("finished_utc", "") >= last.get("finished_utc", ""):
            continue
        record = {key: raw.get(key) for key in (
            "item_id", "messages_sha256", "input_context_sha256", "invocation_id", "prior_attempts")}
        record.update(status=last["status"], prediction=None, usage=last.get("usage", {}),
                      attempts=len(attempts), cumulative_attempts=raw.get("prior_attempts", 0) + len(attempts),
                      started_utc=attempts[0]["started_utc"], finished_utc=last["finished_utc"],
                      recovered_from_persisted_raw=True, latency_source="persisted raw attempt interval")
        record["latency_seconds"] = max(0, (datetime.fromisoformat(last["finished_utc"])
                                            - datetime.fromisoformat(attempts[0]["started_utc"])).total_seconds())
        for key in ("error", "seed", "seed_ordinal", "request_sha256", "finish_reason", "response_id",
                    "diagnostic_consistency", "diagnostic_consistency_error"):
            if key in last:
                record[key] = last[key]
        if last["status"] == "ok":
            record.update(_parse_response(last["response"]))
        predictions[raw["item_id"]] = record
        recovered += 1
    if recovered:
        _rewrite_jsonl(predictions_path, list(predictions.values()))
    return recovered


def _recovery_started(output_dir: Path) -> bool:
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    return any(entry.get("mode") == "retry_failed" for entry in metadata.get("retry_invocations", []))


def _recovery_invocation_ids(output_dir: Path) -> set[str] | None:
    path = output_dir / "run_metadata.json"
    if not path.exists():
        return None
    entries = json.loads(path.read_text()).get("retry_invocations", [])
    start = next((index for index, entry in enumerate(entries) if entry.get("mode") == "retry_failed"), None)
    return None if start is None else {entry["invocation_id"] for entry in entries[start:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--model", required=True, help="The verified adapted model ID exposed by /v1/models")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="QWEN35_LOCAL_API_KEY")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--thinking-override", choices=("true", "false"),
                        help="Explicit mode ablation in a new output directory; all other settings stay frozen")
    parser.add_argument("--resume", action="store_true", help="Resume the exact persisted protocol without replenishing recovery budgets")
    args = parser.parse_args()
    baseline = json.loads((args.baseline_run_dir / "run_metadata.json").read_text())
    training = json.loads(args.training_metadata.read_text())
    server = json.loads(args.server_config.read_text())
    if args.adapter_path.resolve() != (args.training_metadata.resolve().parent / "adapter"):
        parser.error("Adapter path must be the final adapter saved alongside training metadata")
    protocol = validate_protocol(baseline, training, server, input_path=args.input, adapter_path=args.adapter_path)
    evaluation_config = dict(baseline["config"])
    if args.thinking_override is not None:
        evaluation_config["thinking"] = args.thinking_override
        protocol.update(
            experiment="thinking_mode_ablation",
            baseline_evaluation_thinking=baseline["config"]["thinking"],
            evaluation_thinking=args.thinking_override,
            inference_changes={"thinking": {"baseline": baseline["config"]["thinking"],
                                            "current": args.thinking_override}},
        )
    if server.get("evaluated_model_name") != args.model:
        parser.error("Model ID must match the adapted model bound by the verified serving manifest")
    if args.gpu_index != server["gpu_index"] or args.bootstrap_repeats < 0:
        parser.error("GPU index must match the frozen server; bootstrap repeats must be nonnegative")
    protocol.update(
        baseline_run_dir=str(args.baseline_run_dir), training_metadata_sha256=_sha(args.training_metadata),
        baseline_metadata_sha256=_sha(args.baseline_run_dir / "run_metadata.json"),
        serving_manifest_sha256=_sha(args.server_config), adapted_model_id=args.model,
        input_path=str(args.input.resolve()), bootstrap_repeats=args.bootstrap_repeats,
        gpu_index=args.gpu_index, api_key_env=args.api_key_env, env_file=str(args.env_file.resolve()),
        stages=["one first-pass request per item", "failed IDs only: one recovery request plus two structural retries"],
        gold_used_to_select_retries=False, retry_on_semantic_inconsistency=False,
    )
    state_path = args.output_dir / "post_training_protocol.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume or not state_path.exists():
            parser.error("A nonempty directory requires --resume and its persisted protocol")
        state = json.loads(state_path.read_text())
        if state.get("config") != protocol:
            parser.error("Resume hashes/settings differ from the persisted protocol")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        state = {"config": protocol, "stage": "first_pass_started"}
        _write_state(state_path, state)
    stage = state["stage"]
    allowed = _recovery_invocation_ids(args.output_dir) if stage == "recovery_started" else None
    recovered = repair_orphan_responses(args.output_dir, allowed_invocation_ids=allowed)
    if recovered:
        state["predictions_recovered_from_raw"] = state.get("predictions_recovered_from_raw", 0) + recovered
        _write_state(state_path, state)
    if stage == "first_pass_started":
        subprocess.run(runner_command(args, evaluation_config, recovery=False), check=True)
        for filename in ("metrics.json", "predictions.jsonl", "run_metadata.json"):
            shutil.copyfile(args.output_dir / filename, args.output_dir / ("first_pass_" + filename))
        state["stage"] = "first_pass_complete"
        _write_state(state_path, state)
        stage = state["stage"]
    first = json.loads((args.output_dir / "first_pass_metrics.json").read_text())
    if stage == "first_pass_complete":
        if first["failed_predictions"]:
            state["stage"] = "recovery_started"
            _write_state(state_path, state)
            subprocess.run(runner_command(args, evaluation_config, recovery=True), check=True)
            state["stage"] = "recovery_complete"
        else:
            state["stage"] = "complete_without_recovery"
        _write_state(state_path, state)
    elif stage == "recovery_started":
        # A crashed recovery must not start another failed-ID invocation. Normal
        # runner resume keeps persisted failed predictions and resumes missing IDs.
        command = runner_command(args, evaluation_config, recovery=True)
        previously_started = _recovery_started(args.output_dir)
        if previously_started:
            command.remove("--retry-failed")
        subprocess.run(command, check=True)
        state["stage"] = "recovery_complete"
        state["recovery_resumed_without_new_failed_id_budget"] = previously_started
        _write_state(state_path, state)
    elif stage not in {"recovery_complete", "complete_without_recovery"}:
        parser.error("Unknown persisted evaluation stage")
    print(json.dumps({"event": "post_training_evaluation_complete", "output_dir": str(args.output_dir),
                      "metrics_path": str(args.output_dir / "metrics.json")}), flush=True)


if __name__ == "__main__":
    main()
