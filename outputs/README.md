# Saved judge experiments

These directories preserve frozen splits, raw model responses, predictions,
metrics, prompt snapshots, completed training metadata and comparison reports.

| Directory | Purpose |
| --- | --- |
| `judge_bench_v1` | Original scenario holdout, 7,200 train / 800 test; 9B fits and thinking comparison |
| `judge_bench_v2` | Training preparation with weighted thinking supervision |
| `judge_bench_v3` | 7,200 train / 800 held-out judgments within all 400 scenarios; 9B and four-shot runs |
| `judge_bench_v4` | 7,600 train / 400 validation; 9B comparison |
| `judge_bench_v4_qwen35_4b` | 4B comparison using the same v4 split |

JSONL snapshots are stored as `.jsonl.gz` without changing their contents.
Manifest hashes refer to the **uncompressed** bytes. Restore a needed working
file with, for example:

```bash
gzip -dk outputs/judge_bench_v1/train.jsonl.gz
gzip -dk outputs/judge_bench_v1/test.jsonl.gz
gzip -dk outputs/judge_bench_v4/validation.jsonl.gz
```

Generated training chats for v1–v3 were verified byte-for-byte against the
corresponding preparation code before removal. Rebuild them with
`evaluation.judge_bench.prepare_training`, `prepare_training_v2` or
`prepare_training_v3`, using a new output directory. Their original manifests
retain the expected output hashes.

The original v4 training chat is preserved as `training/chat.jsonl.gz` because
rebuilding can introduce tiny floating-point differences in class weights.
It retains the exact training-input hash recorded by both v4 model runs.

Original source code before consolidation is available at commit `7b1e75e`.
Historical job files describe completed runs and their original server paths;
adapt paths and create a fresh job configuration for a new run. Model weights
remain on their training servers.

Historical manifests retain the original source paths, including the former
human-evaluation catalogs that have now been removed. The split snapshots
preserve the complete model inputs and human labels used in the experiments.
Current annotation exports live in `annotation_app/results/`; those CSVs and
the historical split artifacts remain unchanged.

Plain JSONL working copies, per-item resume caches, runtime logs and exploratory
figure folders stay local and are excluded from Git.
