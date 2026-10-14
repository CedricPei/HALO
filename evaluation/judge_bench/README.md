# HALO JudgeBench v1

This release holds out **800** of the **8,000** completed human annotations for judge evaluation and retains **7,200** for training. The two frozen human-evaluation catalogs and the original annotation CSV remain unchanged.

## Split

The builder uses seed `20261010` to select one complete scenario in each of the 40 domain/trigger strata, without consulting annotations or model predictions. Each scenario contributes 20 responses, two from each of ten source models.

| Split | Responses | Scenarios | Per source model | Per domain | Per trigger |
|---|---:|---:|---:|---:|---:|
| JudgeBench | 800 | 40 | 80 | 80 | 200 |
| Training | 7,200 | 360 | 720 | 720 | 1,800 |

Scenarios and complete model inputs do not overlap. User instances can overlap: this evaluates held-out scenarios within the existing instance settings. `split_manifest.json` records selection, distributions, source/output hashes, input duplication and annotation conflicts. Human labels are preserved, including empty diagnostic values normalized to JSON null. Each response has one human annotation; these labels are not adjudicated consensus.

```bash
python -B -m evaluation.judge_bench.dataset --output-dir outputs/judge_bench_v1
```

## Evidence and prompt

Each JSONL row separates `input_context`, `gold`, and `provenance`. Only `input_context` goes to the judge:

- `background`: full Instance Overview, User Profile, and Scenario description, matching the human annotation page as requested.
- `checkpoint_prefix`: all observed events through this checkpoint.
- `retrieval_rounds`: all executed inspection requests and returned results, grouped in their original order.
- `candidate_action`: the final command or null, which is the object being judged.

No evidence is truncated. Scenario descriptions retain their stated objectives, needs, constraints and potential consequences. The judge distinguishes these potential consequences from observed events. Human labels/comments, source-model identity, model reasoning and the action's execution result are not input evidence.

`prompt.py` contains the shared rubric, JSON schema and message builder. The frozen prompt hash covers the instructions, schema and evidence rendering contract. The runner also records the source-file hash and saves `frozen_prompt.json`. Freeze the prompt before running test inference, and reuse it for later models. Transport/sampling settings are explicit run configuration; changing them requires a new run directory.

The training JSONL preserves all input evidence and human verdict, timing, intensity, action level, rejection dimensions, other-issue label and available comments. It supplies the training data without fabricating rationales where humans left comments empty.

## Qwen3.5-9B baseline

Run from the HALO repository root on your inference server. The key is read from the existing `.env` file and is never saved in results. The inference endpoint is available on the server itself.

```bash
python -B -m evaluation.judge_bench.runner \
  --input outputs/judge_bench_v1/test.jsonl \
  --output-dir outputs/judge_bench_v1/runs/qwen3.5-9b \
  --base-url http://127.0.0.1:8003/v1 \
  --model qwen3.5-9b --api-key-env QWEN35_LOCAL_API_KEY \
  --env-file .env --concurrency 64 \
  --thinking true --structured-output true --max-tokens 16384 \
  --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5 \
  --server-config outputs/judge_bench_v1/serving_config.json
```

These Qwen decoding settings follow the [official model card](https://huggingface.co/Qwen/Qwen3.5-9B). The same rubric and input context are reusable with another OpenAI-compatible HTTP(S) endpoint; use `--thinking auto` to omit the Qwen chat-template flag and disable structured output if an endpoint does not implement JSON-schema generation. All configuration and failures remain recorded.

The runner supports resumable inference with the same command. It refuses a different input, prompt or decoding configuration in an existing run directory. Malformed JSON, schema failures and malformed API response shapes receive up to two additional attempts by default (`--max-structure-retries 2`). Transient HTTP/transport failures have a separate bounded budget (`--max-retries 2`). Structurally valid outputs with inconsistent diagnostic labels are scored unchanged and never retried for that inconsistency. Retry decisions do not consult human labels.

`--retry-failed` preserves every successful item and requests only failed or missing IDs. To retain a first-pass baseline, copy its run directory to a separate recovery directory, then run the same command against that copy with `--retry-failed`. Each target receives one recovery request and up to `--max-structure-retries` additional structural retries; historical attempts remain in the copied raw file. This is recovery after an earlier attempt, not an independent first-pass model result.

Messages, context, schema and decoding parameters remain frozen during retries. The original request uses the base seed. Subsequent requests derive a reproducible seed from the base seed, item ID and the number of prior HTTP attempts for that item using the recorded SHA256 seed strategy. Errors are not added to the prompt. Each new raw attempt records its response, usage, seed, request hash, error/status and latency. Recovery policies and runner source hashes are appended under `retry_invocations`; the original `config` is preserved, including when resuming a directory created before structural retries were added.

## Scoring and artifacts

The requested run retains `predictions.jsonl`, `raw_responses.jsonl`, `metrics.json`, `run_metadata.json`, and `frozen_prompt.json`. Raw responses include the API-reported reasoning and usage; labels are never repaired.

Verdict accuracy, macro F1, balanced accuracy, Cohen's kappa and confusion matrices use all 800 items. Missing/structurally invalid outputs count as incorrect. Cross-field diagnostic inconsistencies are reported separately and do not erase an otherwise valid verdict. The strict-consistency verdict score additionally requires consistent diagnostics. Timing, intensity, action-level and rejection-dimension scores evaluate each raw field; intensity's main denominator excludes human null labels, with an additional null-inclusive score. Breakdowns cover domains, triggers and source models. Confidence intervals resample complete scenario groups.

GPU utilization is sampled throughout inference, and run metadata retains utilization distributions, throughput, latency, token usage, served configuration and model-artifact hashes. Throughput and token totals include every raw inference attempt across all sessions, including discarded/failed generations. Request latency and all-attempt finish reasons are separate from `final_item_latency_seconds` and `final_finish_reason_counts` for the final item predictions. Temporary plumbing probes and their logs are removed after verification.

Generate the readable report from the saved run without calling a model:

```bash
python -B -m evaluation.judge_bench.report \
  --run-dir outputs/judge_bench_v1/runs/qwen3.5-9b \
  --output outputs/judge_bench_v1/report.html
```

## Saved experiments

Versioned splits, predictions, raw responses and completed training evidence are
backed up under [`outputs/`](../../outputs/README.md). JSONL snapshots use gzip;
unpack the files needed by a command with `gzip -dk <file.jsonl.gz>`.

The current `annotation_app/results/eval-annotations.csv` is the 8,000-row export with
`E0001`–`E8000` IDs and presentation column names. The historical v1 builder
expects the earlier raw annotation schema and catalog IDs. Use the preserved v1
split snapshots to reproduce the recorded experiments; their original annotations
remain embedded in `gold.human_annotation`.

See [training protocols](TRAINING.md) for the historical fits and the v4
7,600/400 validation comparison. Shared preparation and token supervision live
in `_supervision.py`; versioned modules retain their own policies and settings.
