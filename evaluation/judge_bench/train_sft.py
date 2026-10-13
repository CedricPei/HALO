"""Train an official Qwen3.5 text LoRA adapter from human JudgeBench labels.

Only the declared fit and internal-validation rows are read. A probe performs a real optimizer update in
a temporary output directory that is deleted before this command returns.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import os
import random
import tempfile
import time
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib

from . import prompt


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def kernel_dispatch(function: Any) -> dict[str, Any]:
    """Read the official dispatch wrapper's selected package fallback, without patching it."""
    queue = [(function, 0)]
    visited: set[int] = set()
    while queue:
        current, depth = queue.pop(0)
        if id(current) in visited or depth > 6 or not inspect.isfunction(current):
            continue
        visited.add(id(current))
        closed = inspect.getclosurevars(current).nonlocals
        implementation = closed.get("implementation")
        if callable(implementation):
            return {
                "implementation_module": getattr(implementation, "__module__", None),
                "implementation_name": getattr(implementation, "__qualname__", type(implementation).__name__),
                "package_fallback_selected": closed.get("is_new_implementation"),
                "hub_kernel_requested": False,
            }
        queue.extend((value, depth + 1) for value in closed.values() if inspect.isfunction(value))
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            queue.append((wrapped, depth + 1))
    return {"implementation_module": None, "implementation_name": None,
            "package_fallback_selected": None, "hub_kernel_requested": False}


def encode_example(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Align Unicode character supervision boundaries to complete tokenizer tokens."""
    if row.get("schema_version") == 3:
        from .supervision_v3 import encode_example as encode_v3
        return encode_v3(row, tokenizer, max_length)
    if row.get("schema_version") == 2:
        from .supervision_v2 import encode_example as encode_v2
        return encode_v2(row, tokenizer, max_length)
    messages, completion = row["messages"], row["completion"]
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("Each training row must have exactly system/user/assistant messages")
    if messages[0]["content"] != prompt.SYSTEM_PROMPT or messages[2]["content"] != completion:
        raise ValueError("Training row does not match the frozen prompt or assistant completion")
    if set(json.loads(messages[1]["content"])) != {
        "background", "checkpoint_prefix", "retrieval_rounds", "candidate_action"
    }:
        raise ValueError("Training user message has unexpected evidence fields")
    policy = row["supervision"]
    boundary = policy["loss_prefix_end"]
    if not isinstance(boundary, int) or not 0 < boundary <= len(completion):
        raise ValueError("Invalid completion character supervision boundary")
    supervised_rationale = policy["rationale_supervised"]
    if supervised_rationale != (not policy["mask_completion_suffix"]):
        raise ValueError("Contradictory rationale masking policy")
    if supervised_rationale and boundary != len(completion):
        raise ValueError("A human rationale must supervise the full assistant completion")
    # Use the official generation prefix. No prompt/template/thinking token has loss.
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    tokenized = tokenizer(
        prefix + completion, add_special_tokens=False, return_offsets_mapping=True,
        truncation=False,
    )
    ids = list(tokenized["input_ids"])
    start, end = len(prefix), len(prefix) + boundary
    labels = [
        token if left >= start and right <= end and right > left else -100
        for token, (left, right) in zip(ids, tokenized["offset_mapping"])
    ]
    # Qwen's assistant terminator differs from the generic end-of-text token.
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_id is None or end_id == tokenizer.unk_token_id:
        raise ValueError("Official Qwen tokenizer lacks the assistant <|im_end|> terminator")
    ids.append(end_id)
    labels.append(end_id if supervised_rationale else -100)
    if len(ids) > max_length:
        raise ValueError(f"{row['item_id']}: {len(ids)} tokens exceeds --max-sequence-length {max_length}; evidence was not truncated")
    label_count = sum(label != -100 for label in labels)
    if not label_count:
        raise ValueError(f"{row['item_id']}: no supervised tokens remain")
    return {
        "input_ids": array("I", ids), "labels": array("i", labels), "length": len(ids),
        "supervised_tokens": label_count, "item_id": row["item_id"],
    }


class EncodedDataset:
    def __init__(self, examples: list[dict[str, Any]]):
        self.examples = examples
        self.lengths = [example["length"] for example in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        result = {"input_ids": list(example["input_ids"]), "labels": list(example["labels"])}
        if "loss_weights" in example:
            result["loss_weights"] = list(example["loss_weights"])
        return result


def length_bucket_indices(lengths: list[int], seed: int, epoch: int, bucket_size: int) -> list[int]:
    """Shuffle every example once, bucket by length, and expose the longest batch first."""
    indices = list(range(len(lengths)))
    random.Random(seed + epoch).shuffle(indices)
    for start in range(0, len(indices), bucket_size):
        indices[start : start + bucket_size] = sorted(
            indices[start : start + bucket_size], key=lambda index: lengths[index], reverse=True
        )
    if indices:
        longest = max(range(len(indices)), key=lambda position: lengths[indices[position]])
        indices[0], indices[longest] = indices[longest], indices[0]
    return indices


class MaskedCollator:
    def __init__(self, pad_token_id: int, pad_multiple: int = 8):
        self.pad_token_id, self.pad_multiple = pad_token_id, pad_multiple

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = math.ceil(max(len(row["input_ids"]) for row in examples) / self.pad_multiple) * self.pad_multiple
        batch = {
            "input_ids": torch.tensor([
                row["input_ids"] + [self.pad_token_id] * (width - len(row["input_ids"])) for row in examples
            ], dtype=torch.long),
            "labels": torch.tensor([
                row["labels"] + [-100] * (width - len(row["labels"])) for row in examples
            ], dtype=torch.long),
            "attention_mask": torch.tensor([
                [1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"])) for row in examples
            ], dtype=torch.long),
        }
        if any("loss_weights" in row for row in examples):
            if not all("loss_weights" in row for row in examples):
                raise ValueError("A microbatch cannot mix supervision protocols")
            batch["loss_weights"] = torch.tensor([
                row["loss_weights"] + [0.0] * (width - len(row["loss_weights"])) for row in examples
            ], dtype=torch.float32)
        return batch


def _length_summary(examples: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(example["length"] for example in examples)
    return {
        "items": len(examples), "input_tokens": sum(lengths),
        "supervised_tokens": sum(example["supervised_tokens"] for example in examples),
        "min": lengths[0] if lengths else None, "max": lengths[-1] if lengths else None,
        "mean": sum(lengths) / len(lengths) if lengths else None,
        "p50": lengths[round((len(lengths) - 1) * 0.5)] if lengths else None,
        "p95": lengths[round((len(lengths) - 1) * 0.95)] if lengths else None,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-chat", type=Path, default=Path("outputs/judge_bench_v1/training/chat.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/judge_bench_v1/training/training_manifest.json"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model-manifest", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cuda-memory-limit-gib", type=float, help="Cap this process PyTorch CUDA allocator when sharing a GPU")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-sequence-length", type=int, default=16384)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--train-all", action="store_true", help="Fixed final refit on all prepared rows; no internal validation")
    parser.add_argument("--supervision-protocol", choices=("legacy-v1", "thinking-weighted-v2", "nonthinking-weighted-v3"), default="legacy-v1")
    parser.add_argument("--probe", action="store_true", help="Real short update(s), temporary output, no saved checkpoint or retained logs")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--tokenize-only", action="store_true", help="Validate all masks/lengths without loading model weights")
    parser.add_argument("--allow-reference-delta", action="store_true", help="Explicitly allow missing FLA and the slow reference DeltaNet path")
    args = parser.parse_args()
    positive = (args.epochs, args.batch_size, args.eval_batch_size, args.gradient_accumulation_steps,
                args.learning_rate, args.lora_r, args.lora_alpha, args.max_sequence_length,
                args.logging_steps, args.eval_steps, args.save_total_limit)
    if any(value <= 0 for value in positive) or not 0 <= args.lora_dropout < 1 or not 0 <= args.warmup_ratio <= 1:
        parser.error("Training sizes/rates must be positive, dropout in [0,1), warmup in [0,1]")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be positive or -1")
    if not args.model_path.is_dir():
        parser.error("--model-path must be the existing official local checkpoint directory")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Select a single physical GPU before importing Torch/Transformers. GPU 1 is
    # never enlisted through Trainer's automatic DataParallel discovery.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    elif visible != str(args.gpu_index):
        raise ValueError("CUDA_VISIBLE_DEVICES must name exactly the declared --gpu-index")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoTokenizer, Qwen3_5ForConditionalGeneration, Trainer, TrainerCallback,
        TrainingArguments, set_seed,
    )
    from .runner import GPUMonitor

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    weighted = args.supervision_protocol != "legacy-v1"
    thinking = args.supervision_protocol == "thinking-weighted-v2"
    row_schema = 2 if thinking else 3 if weighted else 1
    validation_policy = manifest.get("internal_validation", {})
    if weighted:
        if manifest.get("training_protocol") != args.supervision_protocol:
            raise ValueError("Weighted training requires its matching supervision manifest")
        if validation_policy.get("enabled") is True:
            if args.train_all:
                raise ValueError("--train-all would consume the declared validation holdout")
        elif validation_policy.get("enabled") is not False or not args.train_all:
            raise ValueError("A weighted no-validation manifest requires --train-all")
    if not weighted and manifest.get("schema_version") != 1:
        raise ValueError("Legacy supervision requires a version-1 training manifest")
    input_sha = _sha(args.input_chat)
    if manifest.get("test_set_used") is not False or manifest["prompt"]["sha256"] != prompt.PROMPT_SHA256:
        raise ValueError("Training manifest uses test data or differs from the frozen judge prompt")
    if manifest["output"]["sha256"] != input_sha:
        raise ValueError("Prepared training JSONL no longer matches its manifest")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=False, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast official tokenizer is required for exact character-boundary loss masking")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, validation = [], []
    rationale_sources: Counter[str] = Counter()
    seen_ids: set[str] = set()
    split_scenarios: dict[str, set[str]] = {"train": set(), "validation": set()}
    split_inputs: dict[str, set[str]] = {"train": set(), "validation": set()}
    split_counts: Counter[str] = Counter()
    tokenization_started = time.perf_counter()
    with args.input_chat.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("schema_version") != row_schema:
                raise ValueError("Training row differs from the declared supervision protocol")
            if row["item_id"] in seen_ids or row["training_split"] not in split_scenarios:
                raise ValueError("Duplicate training item ID or invalid split")
            seen_ids.add(row["item_id"])
            split_scenarios[row["training_split"]].add(row["provenance"]["scenario_key"])
            split_inputs[row["training_split"]].add(row["hashes"]["source_input_context_sha256"])
            split_counts[row["training_split"]] += 1
            encoded = encode_example(row, tokenizer, args.max_sequence_length)
            rationale_sources[row["supervision"]["rationale_source"]] += 1
            (validation if row["training_split"] == "validation" and not args.train_all else fit).append(encoded)
    if len(seen_ids) != manifest["rows"]:
        raise ValueError("Training row count differs from the manifest")
    if split_inputs["train"] & split_inputs["validation"]:
        raise ValueError("An identical full input crosses the training/validation split")
    if (split_scenarios["train"] & split_scenarios["validation"]
            and validation_policy.get("allow_shared_scenarios") is not True):
        raise ValueError("Internal validation scenarios overlap fit scenarios without an item-holdout policy")
    if weighted and any(split_counts[name] != manifest["distributions"][name]["items"]
                        for name in ("train", "validation")):
        raise ValueError("Fit/validation counts differ from the weighted manifest")
    if not fit or (not args.train_all and not validation):
        raise ValueError("The fit/internal-validation split is empty")
    summary = {
        "event": "tokenization_complete", "fit": _length_summary(fit),
        "validation": _length_summary(validation), "rationale_sources": dict(rationale_sources),
        "wall_seconds": time.perf_counter() - tokenization_started, "evidence_truncated": False,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.tokenize_only:
        return summary
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A single visible CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training configuration requires CUDA BF16 support")
    if _version("flash-linear-attention") is None and not args.allow_reference_delta:
        raise RuntimeError("Official flash-linear-attention is required; install it before using the slow reference DeltaNet")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_implementation

    kernels = {
        "delta_rule": kernel_dispatch(qwen_implementation.torch_chunk_gated_delta_rule),
        "causal_conv1d": kernel_dispatch(qwen_implementation.causal_conv1d_fn),
    }
    print(json.dumps({"event": "kernel_dispatch", **kernels}), flush=True)
    if not (kernels["delta_rule"]["implementation_module"] or "").startswith("fla.") and not args.allow_reference_delta:
        raise RuntimeError("The official Qwen DeltaNet wrapper did not select FLA; inspect installed dependencies before training")
    torch.cuda.set_device(0)
    if args.cuda_memory_limit_gib is not None:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        limit_bytes = args.cuda_memory_limit_gib * 1024**3
        if not 0 < limit_bytes < total_bytes:
            raise ValueError("CUDA allocator limit must be positive and below GPU capacity")
        torch.cuda.set_per_process_memory_fraction(limit_bytes / total_bytes, device=0)
        print(json.dumps({"event": "cuda_allocator_limit", "limit_gib": args.cuda_memory_limit_gib}), flush=True)
    torch.set_float32_matmul_precision("high")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model_path), dtype=torch.bfloat16, trust_remote_code=False,
        local_files_only=True, attn_implementation=args.attn_implementation,
        device_map={"": 0},
    )
    text_prefix = "model.language_model."
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and name.startswith(text_prefix)]
    if not targets:
        raise RuntimeError("Official Qwen3.5 text-backbone linear module names were not found")
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=targets, bias="none",
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if any(parameter.requires_grad for name, parameter in model.named_parameters()
           if ".lora_" not in name):
        raise RuntimeError("An unexpected base/vision parameter is trainable")

    class SupervisedPositionTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            if dataset is None:
                return None

            class LengthBucketSampler(torch.utils.data.Sampler):
                def __init__(self):
                    self.epoch = 0

                def __len__(self):
                    return len(dataset)

                def set_epoch(self, epoch):
                    self.epoch = epoch

                def __iter__(self):
                    return iter(length_bucket_indices(
                        dataset.lengths, args.seed, self.epoch,
                        args.batch_size * args.gradient_accumulation_steps * 50,
                    ))

            return LengthBucketSampler()

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            shifted = labels[:, 1:]
            positions = torch.nonzero((shifted != -100).any(dim=0), as_tuple=True)[0]
            if positions.numel() == 0:
                raise ValueError("The collated microbatch has no causal supervised labels")
            outputs = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                use_cache=False, logits_to_keep=positions,
            )
            # Compute exactly the same masked causal CE as full logits, avoiding
            # the large vocabulary projection for every prompt position.
            selected_labels = shifted[:, positions]
            if weighted:
                selected_weights = inputs["loss_weights"][:, 1:][:, positions]
                token_loss = F.cross_entropy(
                    outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                    selected_labels.reshape(-1), ignore_index=-100, reduction="none",
                ).reshape_as(selected_weights)
                loss = (token_loss * selected_weights).sum() / selected_weights.sum()
            else:
                loss = F.cross_entropy(
                    outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                    selected_labels.reshape(-1), ignore_index=-100,
                )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite supervised causal loss")
            return (loss, outputs) if return_outputs else loss

    class ProgressCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            print(json.dumps({"event": "training_started", "max_steps": state.max_steps}), flush=True)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                print(json.dumps({"event": "first_optimizer_step_complete", "step": state.global_step,
                                  "epoch": state.epoch,
                                  "cuda_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                                  "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}), flush=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            print(json.dumps({"event": "training_progress", "step": state.global_step,
                              "epoch": state.epoch, **(logs or {}),
                              "cuda_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                              "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}), flush=True)

        def on_epoch_end(self, training_args, state, control, **kwargs):
            if validation and not args.probe:
                control.should_evaluate = True
            if not args.no_save:
                control.should_save = True
            return control

    expected_steps = math.ceil(math.ceil(len(fit) / args.batch_size) / args.gradient_accumulation_steps) * args.epochs
    effective_max_steps = args.max_steps if args.max_steps > 0 else math.ceil(expected_steps)
    warmup_steps = math.ceil(args.warmup_ratio * effective_max_steps)
    config = {
        "input_sha256": input_sha, "manifest_sha256": _sha(args.manifest),
        "prompt_sha256": prompt.PROMPT_SHA256, "model_path": str(args.model_path.resolve()),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, "max_sequence_length": args.max_sequence_length,
        "warmup_ratio": args.warmup_ratio, "warmup_steps": warmup_steps,
        "weight_decay": args.weight_decay,
        "seed": args.seed, "train_all": args.train_all, "max_steps": args.max_steps,
        "effective_max_steps": effective_max_steps,
        "base_dtype": "bfloat16", "optimizer": "adamw_torch_fused", "scheduler": "cosine",
        "attn_implementation": args.attn_implementation, "chat_template_enable_thinking": thinking,
        "supervision_protocol": args.supervision_protocol,
        "mask_policy": ("Official thinking=false prefix masked; label values, JSON structure and assistant end supervised; only absent/long rationale contents masked."
                        if weighted and not thinking else "Official thinking=true prefix masked; thinking closure, JSON structure and assistant end supervised; only absent/long rationale contents masked."
                        if weighted else "Prompt and partial boundary tokens masked; human-comment completion/EOS supervised, absent/long rationale suffix and EOS masked."),
        "logits_policy": ("Preceding supervised positions only; class-weighted per-field token-length-normalized causal CE."
                          if weighted else "Only preceding positions of supervised labels; exact masked causal cross entropy."),
        "class_balance": manifest.get("class_balance") if weighted else None,
        "training_scenarios": manifest.get("distributions", {}).get("train", {}).get("scenarios"),
        "sampler": "Seeded per-epoch length buckets; every fit item exactly once, longest example first",
        "length_bucket_size": args.batch_size * args.gradient_accumulation_steps * 50,
        "test_set_used": False,
    }
    if args.cuda_memory_limit_gib is not None:
        config["cuda_memory_limit_gib"] = args.cuda_memory_limit_gib
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "training_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["config"] != config or not args.resume_from_checkpoint:
            raise ValueError("Existing training directory requires an explicit compatible checkpoint resume")
    metadata = {
        "started_utc": _utc(), "config": config, "data": summary,
        "status": "running", "probe": args.probe, "checkpoints_saved": not args.no_save,
        "trainer_source_sha256": _sha(Path(__file__)), "target_modules": targets,
        "supervision_source_sha256": _sha(Path(__file__).with_name("supervision_v2.py" if thinking else "supervision_v3.py")) if weighted else None,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "trainable_dtypes": dict(Counter(str(p.dtype) for p in model.parameters() if p.requires_grad)),
        "dependencies": {name: _version(name) for name in (
            "torch", "transformers", "peft", "accelerate", "flash-linear-attention", "causal-conv1d", "triton")},
        "kernel_dispatch": kernels,
        "device": {"physical_gpu_index": args.gpu_index, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                   "name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))},
        "expected_epoch_steps": math.ceil(expected_steps),
        "resume_from_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
    }
    if args.base_model_manifest:
        metadata["base_model_artifacts"] = json.loads(args.base_model_manifest.read_text(encoding="utf-8"))
    if not args.no_save:
        _write_json(metadata_path, metadata)
    has_validation = bool(validation) and not args.probe
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs, max_steps=effective_max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, warmup_steps=warmup_steps,
        lr_scheduler_type="cosine", optim="adamw_torch_fused", bf16=True, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0, logging_steps=args.logging_steps, logging_strategy="steps",
        logging_nan_inf_filter=False,
        eval_strategy="steps" if has_validation else "no", eval_steps=args.eval_steps,
        save_strategy="no" if args.no_save else "steps", save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=has_validation and not args.no_save,
        metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=True,
        report_to=[], disable_tqdm=True, seed=args.seed, data_seed=args.seed,
    )
    trainer = SupervisedPositionTrainer(
        model=model, args=training_args, train_dataset=EncodedDataset(fit),
        eval_dataset=EncodedDataset(validation) if has_validation else None,
        data_collator=MaskedCollator(tokenizer.pad_token_id), processing_class=tokenizer,
        callbacks=[ProgressCallback()],
    )
    # The custom token-mean loss is normalized once by Trainer's accumulation,
    # rather than claiming to consume its num_items_in_batch loss kwarg.
    trainer.model_accepts_loss_kwargs = False
    monitor = GPUMonitor(args.gpu_index, 1.0, True)
    probe_parameter = None
    probe_before = None
    if args.probe:
        probe_parameter = next((name, parameter) for name, parameter in model.named_parameters()
                               if parameter.requires_grad and ".lora_B." in name)
        probe_before = probe_parameter[1].detach().flatten()[:1024].clone()
    monitor.start()
    started = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
        metadata["train_metrics"] = result.metrics
        if args.probe:
            after = probe_parameter[1].detach().flatten()[:1024]
            changed = int(torch.count_nonzero(after != probe_before).item())
            metadata["probe_optimizer_update"] = {
                "parameter": probe_parameter[0], "sampled_values": after.numel(),
                "changed_values": changed, "verified": changed > 0,
            }
            if not changed:
                raise RuntimeError("Probe optimizer did not change the sampled LoRA B parameter")
        if has_validation:
            metadata["validation_metrics"] = trainer.evaluate(metric_key_prefix="internal_validation")
        metadata["global_step"] = trainer.state.global_step
        metadata["epochs_completed"] = trainer.state.epoch
        metadata["best_internal_validation_checkpoint"] = trainer.state.best_model_checkpoint
        complete = not args.probe and trainer.state.global_step >= math.ceil(expected_steps)
        metadata["status"] = "complete" if complete else "probe_complete" if args.probe else "partial_complete"
        metadata["full_epoch_training_complete"] = complete
        metadata["ready_for_judgebench_evaluation"] = complete and not args.no_save
        if not args.no_save:
            trainer.save_model(str(args.output_dir / "adapter"))
            tokenizer.save_pretrained(str(args.output_dir / "adapter"))
            trainer.save_state()
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}",
                        global_step=trainer.state.global_step)
        raise
    finally:
        metadata["finished_utc"] = _utc()
        metadata["active_wall_seconds"] = time.perf_counter() - started
        metadata["gpu"] = monitor.finish()
        metadata["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        if not args.no_save:
            _write_json(metadata_path, metadata)
        print(json.dumps({"event": "training_finished", **metadata}, ensure_ascii=False), flush=True)
    return metadata


def main() -> None:
    args = _arguments()
    if args.probe:
        args.no_save = True
        # A one-step warmup would otherwise use LR=0 and perform no weight update.
        args.warmup_ratio = 0.0
        if args.max_steps < 0:
            args.max_steps = 1
        with tempfile.TemporaryDirectory(prefix="judge-sft-probe-") as directory:
            args.output_dir = Path(directory)
            run(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
