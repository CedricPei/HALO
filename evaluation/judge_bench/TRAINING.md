# JudgeBench Qwen3.5 SFT

## Current 7,600 / 400 run

Version 4 (`prepare_training_v4.py`) repartitions the original 8,000 annotations into 7,600 training and 400 validation items. Each of the 400 known scenarios contributes 19 training items and one validation item; each domain/trigger stratum holds out one item per source model. Selection ignores labels, and exact complete-input duplicates remain together. Historical splits, model adapters, and results are preserved.

The 400 rows are validation data for checkpoint selection, not a separate test set. Only the 7,600 training labels determine class weights. Training keeps the non-thinking weighted supervision protocol, original prompt, and unchanged human labels. The run starts from the original Qwen3.5-9B base for one epoch (475 updates), with learning rate 5e-5, LoRA rank 32 / alpha 64, and batch 8 / accumulation 2. Validation loss is evaluated every 100 steps and at epoch end; `adapter/` receives the best validation checkpoint.

The persistent CPU supervisor `lora_job.py` reads `outputs/judge_bench_v4/training/launcher.json`. It waits for three idle GPU1 observations, verifies the GPU UUID and frozen input/code hashes, and prevents duplicate launches with a job lock. It does not stop existing services. Status is saved in `training/qwen3.5-9b-lora/job.json`; once running, training metrics and checkpoints are saved beside it. Cancel this supervisor by its recorded PID to cancel this job only.

## Historical runs

The historical version-3 run uses `resplit.py` and `prepare_training_v3.py`. It holds out two items per scenario, leaving 18 training items in each of all 400 scenarios (7,200 train / 800 test). Exact complete-input duplicates stay together; human labels and comments are unchanged. Selection is label-blind and balances source models inside each domain/trigger stratum. This measures held-out judgments within the known benchmark scenarios. The original scenario holdout remains preserved under `outputs/judge_bench_v1`.

Version 3 trains all 7,200 items with no internal validation and evaluates the fixed final checkpoint. Training and inference both use the official `enable_thinking=false` template and the original judge prompt. Field-normalized label loss uses inverse-square-root class weights capped at a 5:1 ratio, computed only from training labels; the main verdict/timing/intensity fields each receive a loss budget of 3, existing human explanations 0.1, and JSON structure / assistant termination 1 each. Missing explanation placeholders are masked without masking JSON completion or EOS. Every item occurs once per epoch, with no minority duplication. The fixed run uses two epochs, learning rate 5e-5, LoRA rank 32 / alpha 64, and batch 8 × accumulation 2 on GPU0.

Before training, the unchanged original base is evaluated on the new 800 items with the same non-thinking decoding and structure-recovery protocol. The old fitted model is not used as a clean baseline because it has already trained on some newly selected holdout items. Training starts from the original base, not the previous adapter. CPU merge, serving restoration, and the final matched comparison run through the persistent supervisors.

Both version 3 serving runs use a 0.95 GPU-memory allocation with concurrency 64 and a 16,384-token prefill batch. The initial base attempt at 0.99 exhausted temporary prefill memory and produced no valid judgments. That entire attempt is preserved separately as `runs/qwen3.5-9b-base-aborted-oom`; it is not used as the comparison baseline. Four long training-only inputs passed the service check before the fresh base run.

Historical version 1 used `prepare_training.py`, fitting 6,840 rows and evaluating 360 rows from 18 internal validation scenarios. Historical version 2 preparation was not launched. Its thinking-compatible draft does not apply to the current non-thinking protocol.

## Historical version 1 implementation and commands

All six human diagnostic fields and their JSON structure have loss. Existing human comments of at most 1,000 characters provide weak rationale supervision. Missing or longer comments are preserved in the data lineage; a placeholder fills the training sequence, but its value, closing syntax and EOS are masked completely. Prompt tokens and tokens crossing the supervision boundary also have no loss. No model-written rationale or chain of thought becomes a human training target.

The trainer uses the official Qwen tokenizer's `enable_thinking=False` generation prefix. The whole prefix, including any empty thinking template, is masked. This is direct-judgment SFT; a future evaluation should record its thinking setting and use a matching base-model decoding setting for a fair comparison. It should not describe a nonthinking adapter versus the earlier thinking baseline as identical decoding.

The implementation loads the official `Qwen3_5ForConditionalGeneration` with `trust_remote_code=False` and local checkpoint files. BF16 base weights remain frozen. LoRA targets every text-backbone linear layer, with default rank 32, alpha 64 and dropout 0.05; vision, embeddings and the vocabulary head stay frozen. Non-reentrant gradient checkpointing, SDPA, fused AdamW, cosine scheduling, a 3% warmup and microbatch 2 with accumulation 8 give an effective batch size of 16. It computes vocabulary logits only at positions immediately preceding supervised labels; the masked causal cross entropy is the same as full-sequence loss. Complete context is preserved, and any example exceeding the configured sequence limit causes an explicit error.

Loss is the mean over supervised tokens within each microbatch, followed by Trainer's gradient-accumulation scaling. It is not a pooled token mean across an optimizer batch. A custom sampler shuffles by seed and epoch, sorts length buckets and presents the longest example first; it covers every fit item once per epoch and does not require the `datasets` package or the removed `TrainingArguments.group_by_length` option.

Qwen3.5's DeltaNet benefits from the official `flash-linear-attention` package. When the native `causal_conv1d` extension is unavailable, the official Transformers convolution fallback uses PyTorch CUDA convolution. Verify the installed FLA dispatch and one real backward/update before a full run. Dependencies are installed in a separate training environment, without changing the serving environment. See the [official Transformers architecture documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [official model implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py), and [PEFT LoRA documentation](https://huggingface.co/docs/peft/package_reference/lora).

From the HALO repository root, use the isolated training Python executable and the existing official model directory:

```bash
CUDA_VISIBLE_DEVICES=0 /data/wenqi/model-training/qwen3.5-9b-judge/venv/bin/python -B \
  -m evaluation.judge_bench.train_sft \
  --model-path /path/to/existing/Qwen3.5-9B \
  --output-dir outputs/judge_bench_v1/training/qwen3.5-9b-lora \
  --tokenize-only
```

Replace `--tokenize-only` with `--probe --max-steps 1 --logging-steps 1` for a real optimizer update. The probe tests the full fit set's length grouping, masks, BF16 forward/backward and optimizer. Its temporary output directory and files are deleted automatically, and no probe adapter is retained. GPU serving must be stopped by the operator before the real update.

For a full fit, omit both flags:

```bash
CUDA_VISIBLE_DEVICES=0 /data/wenqi/model-training/qwen3.5-9b-judge/venv/bin/python -B \
  -m evaluation.judge_bench.train_sft \
  --model-path /path/to/existing/Qwen3.5-9B \
  --output-dir outputs/judge_bench_v1/training/qwen3.5-9b-lora \
  --epochs 1 --batch-size 2 --gradient-accumulation-steps 8 \
  --learning-rate 0.0001 --lora-r 32 --lora-alpha 64 \
  --max-sequence-length 16384 --eval-steps 100
```

Internal validation selects checkpoints without consulting JudgeBench labels. Each epoch's final checkpoint is evaluated even when its step is not a multiple of the interval. `adapter/` contains the selected adapter and tokenizer; retained `checkpoint-*` directories include optimizer state for `--resume-from-checkpoint`. `training_metadata.json` records input/prompt hashes, exact configuration, dependency versions, target modules, trainable parameters, validation losses, step/epoch progress and GPU/VRAM summaries. A short `--max-steps` run is marked partial and is not presented as a finished training result. The trainer never launches a JudgeBench evaluation automatically.

For a background production job, run `python -B -m evaluation.judge_bench.training_job --launch-config outputs/judge_bench_v1/training/launch.json` in a new session. The standard-library supervisor reads `command` (an executable argument list), `cwd`, `environment` (only public CUDA/cache/thread options) and `output_dir`. It records `job.json` and checks that the declared `gpu_index`/optional `gpu_uuid` is free before starting training. The training process stays in the supervisor's process group. An optional `temporary_root` contains a newly created private training `TMPDIR`; only that private child directory is deleted when training exits.

Optional serving restoration uses `restore_serving_launcher` (an executable argument list), `restore_pid_file`, `restore_log_file`, `restore_base_url`, `restore_model_id` and optional `restore_model_path`. After training succeeds or fails, the supervisor checks GPU processes again and starts serving in a new session only when the GPU is free and the recorded serving PID is inactive. It verifies `/health` and the declared model ID from `/v1/models` for at most 240 seconds. `restore_env_file` and `restore_api_key_env` name an existing environment file/key; the key remains in memory and is never copied into job metadata. A timed-out serving process remains running for inspection, with its PID and log recorded. Restoration does not launch a JudgeBench evaluation. Existing `job.json` is never overwritten by a new invocation.

When post-training evaluation is requested, `merge_adapter.py` creates a new checkpoint from the completed final adapter using CPU arithmetic only. It adds each declared FP32 LoRA delta to its original BF16 linear weight, preserving all 775 original tensor keys and all 15 MTP tensors. It verifies every serialized tensor, every consumed adapter key, original file hashes and completed training lineage before publishing the new directory atomically. `--resume` only reuses a complete directory whose lineage and all file hashes still match.

```bash
python -B -m evaluation.judge_bench.merge_adapter \
  --base-model /path/to/original/Qwen3.5-9B \
  --adapter outputs/judge_bench_v1/training/qwen3.5-9b-lora/adapter \
  --training-metadata outputs/judge_bench_v1/training/qwen3.5-9b-lora/training_metadata.json \
  --base-manifest outputs/judge_bench_v1/runs/qwen3.5-9b-retried/run_metadata.json \
  --output-dir /path/to/new/Qwen3.5-9B-judge --cpu-threads 4
```

The deployment operator then serves that new path using the original model ID and frozen MTP2/server settings, checks a training-only plumbing example, and records a verified serving manifest. `post_training_eval.py` requires that manifest, the completed training metadata and final adapter. It checks the test/prompt/schema hashes and copies all decoding settings from the preserved baseline, including `thinking=true`, rather than choosing new settings. Its serving manifest includes `evaluated_model_name` and `adaptation` with `method`, `adapter_sha256`, `adapter_config_sha256`, `all_base_keys_preserved` and `mtp_tensors_unchanged` from the merge manifest.

The evaluation performs one 800-item first pass, preserves its metrics/predictions/metadata, and starts one failed-ID recovery invocation only when needed, matching the baseline's two-stage protocol. `--resume` requires identical persisted protocol hashes and settings. Recovery resumes without granting another failed-ID budget once that invocation exists; complete raw responses whose final prediction write was interrupted are recovered without a new request. Training's direct-judgment template and evaluation's original thinking setting remain explicitly recorded.

The requested training-after comparison is a separate durable supervisor: `python -B -m evaluation.judge_bench.post_training_job --config outputs/judge_bench_v1/post_training_launch.json`. It waits for the full epoch, selected final adapter and successful serving restoration before doing anything to the GPU. CPU merging preserves every original safetensors key and the original MTP tensors, with source/adapter/merged hashes recorded. Only this user's verified serving process group on GPU 0 is replaced; checkpoint path changes while every frozen serving flag and the model ID remain the same.

`post_training_eval.py` uses the retried baseline's input, prompt, JSON schema and decoder configuration verbatim, including `thinking=true`, temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, max output 16,384, seed 20261010 and concurrency 64. Thus training uses a direct-judgment template, but both evaluation runs use the same thinking-enabled inference template. First-pass inference is followed only by the original structural-failure recovery policy; semantic errors are never retry signals. Completed evidence is retained and safe resume checks prevent duplicate supervisors and configuration changes.

`compare.py` writes `comparison/report.html` and `comparison/comparison.json` without modifying either run. Four headline metrics are verdict accuracy (800), timing accuracy (800), intensity accuracy on non-null human labels (772), and exact verdict/timing/intensity joint accuracy (800, including null). Per-class precision/recall, confusion matrices, majority-class reference scores and non-default subsets explain imbalance. The original `verdict_with_strict_consistency` is a separate consistency diagnostic, not joint label accuracy. Until the new evaluation completes, after-training scores and changes are explicitly null.

## Qwen3.5-4B on shared GPU memory

The 4B run is stored in `outputs/judge_bench_v4_qwen35_4b`. It reuses the
frozen v4 7,600/400 data, full prompts, supervision weights and seed. The official
`Qwen/Qwen3.5-4B` revision is
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; its artifact manifest records
every file hash and 738 base tensor keys, including 15 MTP keys.

The user authorized sharing unused VRAM. After GPU1's foreign evaluation
allocation grew, the run moved to GPU0 before any training update. It preserves
the existing GPU0 training. Training requires 60 GiB free at admission and caps
this process's PyTorch CUDA allocator at 48 GiB. Microbatch 2 and accumulation 8
retain effective batch size 16 and 475 updates for one epoch. This is a resource
configuration change from the 9B run's microbatch 8 and accumulation 2.

`training/launcher.json` and `training/qwen3.5-4b-lora/job.json` are the
training authority. `validation_launch.json` and `validation_job/job.json`
control CPU merging and the LoRA/original-model 400-item comparison. Serving
uses localhost:8014, GPU0, vLLM memory fraction 0.45 and a fixed 20 GiB KV cache.
Inference and structural recovery otherwise match the 9B v4 protocol. Per-request
latency is neither measured nor saved.

Do not relaunch a live controller. Inspect its UID, process group, start ticks,
lock and children first. Shared service cleanup verifies and signals only the
job's own process group, then waits for that group's GPU allocations to disappear.
Foreign allocations are permitted and retained. These 400 items select the
checkpoint and share scenarios with training, so the comparison is validation,
not an independent test of generalization.
