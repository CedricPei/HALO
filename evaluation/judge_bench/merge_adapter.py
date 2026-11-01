"""Merge a completed text LoRA into original safetensors on CPU, preserving MTP.

No Transformers model is instantiated. Every original key is copied, every
adapter key is consumed exactly once, and serialized tensors are verified before
the new model directory is published. Original files are never modified.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
from typing import Any


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifacts(path: Path) -> dict[str, Any]:
    return {file.name: {"bytes": file.stat().st_size, "sha256": _sha(file)}
            for file in sorted(path.iterdir()) if file.is_file() and file.name != "merge_manifest.json"}


def _tensor_sha(tensor: Any) -> str:
    import torch
    return hashlib.sha256(memoryview(tensor.contiguous().reshape(-1).view(torch.uint8).numpy())).hexdigest()


def normalize_target_selectors(selectors: Any, full_targets: list[str]) -> dict[str, list[str]]:
    """Expand PEFT's saved exact/suffix selectors against declared full targets.

    PEFT may reduce hundreds of explicit target names to equivalent leaf names
    when saving its config. The actual A/B tensor names are still checked against
    the complete original target set separately; selector normalization cannot
    add, drop or rename a tensor target.
    """
    if not isinstance(selectors, list) or not selectors or any(not isinstance(name, str) or not name for name in selectors):
        raise ValueError("Adapter target_modules must be a nonempty explicit selector list")
    if len(selectors) != len(set(selectors)) or len(full_targets) != len(set(full_targets)):
        raise ValueError("Adapter selectors or declared full targets contain duplicates")
    expanded = {}
    covered = set()
    for selector in selectors:
        matches = sorted(name for name in full_targets if name == selector or name.endswith("." + selector))
        if not matches:
            raise ValueError(f"Adapter target selector has no declared trained module: {selector}")
        expanded[selector] = matches
        covered.update(matches)
    if covered != set(full_targets):
        raise ValueError("Adapter target selectors do not cover every completed training target")
    return expanded


def merge(args: argparse.Namespace) -> dict[str, Any]:
    # This process is CPU-only even when launched by a GPU-serving supervisor.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    destination = args.output_dir.resolve()
    if destination.is_relative_to(args.base_model.resolve()) or destination.is_relative_to(args.adapter.resolve()):
        raise ValueError("Merged output must be outside the original base and adapter directories")
    training = json.loads(args.training_metadata.read_text())
    if not (training.get("status") == "complete" and training.get("full_epoch_training_complete") is True
            and training.get("ready_for_judgebench_evaluation") is True):
        raise ValueError("Only the final adapter from completed full-epoch training may be merged")
    if args.adapter.resolve() != args.training_metadata.resolve().parent / "adapter":
        raise ValueError("Adapter path must be the final training adapter")
    if training["config"].get("model_path") != str(args.base_model.resolve()):
        raise ValueError("Merge base differs from the checkpoint used to train")
    base_evidence = json.loads(args.base_manifest.read_text())
    base_evidence = base_evidence.get("served_configuration", base_evidence)
    if base_evidence.get("model_path") != str(args.base_model.resolve()):
        raise ValueError("Baseline manifest points at another base checkpoint")
    base_artifacts = _artifacts(args.base_model)
    for name, evidence in base_evidence["model_artifacts"].items():
        if base_artifacts.get(name) != evidence:
            raise ValueError(f"Original base artifact mismatch: {name}")
    training_artifacts = training.get("base_model_artifacts", {}).get("model_artifacts")
    lineage = {"method": "training_metadata_full_base_artifacts"}
    if training_artifacts != base_evidence["model_artifacts"]:
        launch_path = getattr(args, "training_launch_config", None)
        if training_artifacts is not None or launch_path is None:
            raise ValueError("Training base artifact lineage differs from baseline")
        launch = json.loads(launch_path.read_text())
        job = json.loads((args.training_metadata.parent / "job.json").read_text())
        if job.get("status") != "completed" or job.get("config_sha256") != _sha(launch_path):
            raise ValueError("Training launch is not the immutable launch of the completed job")
        frozen = launch.get("frozen_sha256", {})
        for filename in ("config.json", "model.safetensors.index.json"):
            model_file = args.base_model.resolve() / filename
            if frozen.get(str(model_file)) != base_artifacts[filename]["sha256"]:
                raise ValueError("Training launch base config/index differs from audited original weights")
        if launch.get("data_counts", {}).get("train") != training["data"]["fit"]["items"]:
            raise ValueError("Training data count differs from the frozen launch")
        lineage = {"method": "frozen_training_launch_config_index_and_preexisting_full_base_artifact_evidence",
                   "training_launch_config_sha256": _sha(launch_path),
                   "verification_timing": "Full original weight hashes verified after training against the preexisting frozen base manifest; training metadata remains unchanged"}

    adapter_file = args.adapter / "adapter_model.safetensors"
    config_file = args.adapter / "adapter_config.json"
    adapter_sha, config_sha = _sha(adapter_file), _sha(config_file)
    config = json.loads(config_file.read_text())
    unsupported = {"fan_in_fan_out": False, "use_dora": False, "use_rslora": False, "lora_bias": False}
    if any(config.get(name, expected) != expected for name, expected in unsupported.items()):
        raise ValueError("This merger supports the declared standard linear LoRA only")
    if config.get("bias", "none") != "none" or any(config.get(name) for name in (
        "modules_to_save", "rank_pattern", "alpha_pattern", "trainable_token_indices", "exclude_modules",
    )):
        raise ValueError("Adapter contains extra supervised parameters or nonuniform LoRA settings")
    rank, alpha = config["r"], config["lora_alpha"]
    if rank != training["config"]["lora_r"] or alpha != training["config"]["lora_alpha"]:
        raise ValueError("Adapter rank/scaling differs from training configuration")
    index = json.loads((args.base_model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    mtp_keys = {name for name in weight_map if name.startswith("mtp.")}
    if len(weight_map) != args.expected_base_keys or len(mtp_keys) != args.expected_mtp_keys:
        raise ValueError("Original base/MTP key counts differ from the audited checkpoint")
    targets = set(training["target_modules"])
    normalized_selectors = normalize_target_selectors(config["target_modules"], training["target_modules"])
    pairs: dict[str, dict[str, Any]] = defaultdict(dict)
    with safe_open(adapter_file, framework="pt", device="cpu") as handle:
        adapter_keys = set(handle.keys())
        for name in adapter_keys:
            prefix = "base_model.model."
            if not name.startswith(prefix):
                raise ValueError(f"Unrecognized adapter prefix: {name}")
            clean = name[len(prefix):]
            endings = {".lora_A.weight": "A", ".lora_B.weight": "B"}
            for ending, side in endings.items():
                if clean.endswith(ending):
                    target = clean[:-len(ending)]
                    break
            else:
                raise ValueError(f"Unsupported adapter tensor: {name}")
            if target not in targets or not target.startswith("model.language_model."):
                raise ValueError(f"Undeclared or nontext adapter target: {target}")
            if side in pairs[target]:
                raise ValueError("Duplicate adapter matrix")
            tensor = handle.get_tensor(name)
            if tensor.dtype != torch.float32 or tensor.ndim != 2 or not torch.isfinite(tensor).all().item():
                raise ValueError(f"Adapter matrix must be finite FP32 linear LoRA: {name}")
            pairs[target][side] = tensor
    if set(pairs) != targets or any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Every trained target must have exactly one A/B matrix pair")
    for target, pair in pairs.items():
        if target + ".weight" not in weight_map or pair["A"].shape[0] != rank or pair["B"].shape[1] != rank:
            raise ValueError(f"Base weight or adapter rank mismatch: {target}")
    identity = {"adapter_sha256": adapter_sha, "adapter_config_sha256": config_sha,
                "training_metadata_sha256": _sha(args.training_metadata), "base_model_artifacts": base_artifacts,
                "base_lineage_verification": lineage}
    if args.output_dir.exists():
        if not args.resume or not (args.output_dir / "merge_manifest.json").is_file():
            raise ValueError("Existing merged directory requires --resume and a complete manifest")
        previous = json.loads((args.output_dir / "merge_manifest.json").read_text())
        if previous.get("status") != "complete" or any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("Existing merged model has another lineage or is incomplete")
        if _artifacts(args.output_dir) != previous["model_artifacts"]:
            raise ValueError("Persisted merged model artifacts have changed")
        return previous
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    # A private sibling directory is removed on failure. Final publication is an
    # atomic rename, so serving never sees a partially written checkpoint.
    with tempfile.TemporaryDirectory(prefix="." + args.output_dir.name + ".merge-", dir=args.output_dir.parent) as tmp:
        stage = Path(tmp)
        for source in args.base_model.iterdir():
            if source.is_file() and source.suffix != ".safetensors":
                shutil.copyfile(source, stage / source.name)
        consumed: set[str] = set()
        mtp_hashes = {}
        original_keys: set[str] = set()
        for shard in sorted(set(weight_map.values())):
            tensors, expected_hashes = {}, {}
            with safe_open(args.base_model / shard, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                keys = set(handle.keys())
                if keys != {name for name, file in weight_map.items() if file == shard}:
                    raise ValueError(f"Shard keys do not match the original index: {shard}")
                for name in sorted(keys):
                    original_keys.add(name)
                    value = handle.get_tensor(name)
                    target = name[:-len(".weight")] if name.endswith(".weight") else None
                    if target in pairs:
                        if value.dtype != torch.bfloat16 or value.ndim != 2:
                            raise ValueError(f"Expected BF16 linear base tensor: {name}")
                        a, b = pairs[target]["A"], pairs[target]["B"]
                        if tuple(value.shape) != (b.shape[0], a.shape[1]):
                            raise ValueError(f"Adapter/base shape mismatch: {name}")
                        value = (value.float() + (b @ a) * (alpha / rank)).to(value.dtype)
                        if not torch.isfinite(value).all().item():
                            raise ValueError(f"Merged tensor is not finite: {name}")
                        consumed.add(target)
                    elif name in mtp_keys:
                        mtp_hashes[name] = _tensor_sha(value)
                    expected_hashes[name] = (tuple(value.shape), value.dtype, _tensor_sha(value))
                    tensors[name] = value
                save_file(tensors, stage / shard, metadata=metadata)
            # Verify every serialized tensor, including all untouched vision,
            # embeddings, vocabulary head and MTP tensors, against source/delta.
            with safe_open(stage / shard, framework="pt", device="cpu") as handle:
                if set(handle.keys()) != keys:
                    raise ValueError("Merged serialization lost or added tensor keys")
                for name in keys:
                    value = handle.get_tensor(name)
                    if (tuple(value.shape), value.dtype, _tensor_sha(value)) != expected_hashes[name]:
                        raise ValueError(f"Merged tensor verification failed: {name}")
            del tensors, expected_hashes, value
        if original_keys != set(weight_map) or consumed != targets or set(mtp_hashes) != mtp_keys:
            raise ValueError("Not every base/MTP/adapter key was accounted for")
        if _artifacts(args.base_model) != base_artifacts or _sha(adapter_file) != adapter_sha or _sha(config_file) != config_sha:
            raise ValueError("Original base or adapter changed while merging")
        manifest = {
            "status": "complete", "method": "merged_safetensors", **identity,
            "base_model_path": str(args.base_model.resolve()), "adapter_path": str(args.adapter.resolve()),
            "all_base_keys_preserved": True, "mtp_tensors_unchanged": True,
            "base_key_count": len(weight_map), "mtp_key_count": len(mtp_keys),
            "adapter_keys_consumed": len(adapter_keys), "updated_weight_count": len(consumed),
            "adapter_target_selector_expansion": normalized_selectors,
            "mtp_tensor_sha256": mtp_hashes, "model_artifacts": _artifacts(stage),
            "merge_arithmetic": "FP32(B @ A) * alpha/r + FP32(original BF16 weight), then cast to original BF16",
            "cpu_threads": args.cpu_threads, "cuda_initialized": torch.cuda.is_initialized(),
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "merger_source_sha256": _sha(Path(__file__)),
        }
        if manifest["cuda_initialized"]:
            raise RuntimeError("CPU merge unexpectedly initialized CUDA")
        (stage / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        stage.replace(args.output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-launch-config", type=Path, help="Immutable completed-training launch with frozen base config/index hashes when training metadata omitted full artifact hashes")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--expected-base-keys", type=int, default=775)
    parser.add_argument("--expected-mtp-keys", type=int, default=15)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.expected_base_keys < 1 or args.expected_mtp_keys < 1:
        parser.error("CPU threads and audited tensor counts must be positive")
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Merge interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    manifest = merge(args)
    print(json.dumps({"event": "adapter_merge_complete", "output_dir": str(args.output_dir),
                      "updated_weights": manifest["updated_weight_count"], "mtp_keys": manifest["mtp_key_count"]}), flush=True)


if __name__ == "__main__":
    main()
