# 400-item validation

This package retains the server's **7,600 training / 400 validation** workflow.
There is one validation response per known scenario, with 40 responses from each
of the ten source models. The 400 rows also participate in checkpoint selection.

The original prompt, full model inputs, human labels and non-thinking,
class-balanced supervision are preserved. The earlier 800-item test workflow,
relay experiments and obsolete training variants have been removed.

Restore the frozen validation input:

```bash
gzip -dk outputs/judge_bench_v4/validation.jsonl.gz
```

`runner.py` evaluates only this 400-item validation split through HTTP(S).
Use a fresh output directory and the decoding settings recorded in the saved
`validation_launch.json`; `--retry-failed` retains successful predictions.
`validation_comparison.py` checks matched Base/LoRA settings and regenerates the
comparison from recorded predictions without model calls.

For training, restore `outputs/judge_bench_v4/training/chat.jsonl.gz` and use
`train_sft.py` with its `training_manifest.json`. It accepts only the recorded
non-thinking supervision protocol and preserves the 7,600/400 boundary.
`merge_adapter.py` merges a completed adapter; `lora_job.py` and
`validation_job.py` retain the server's training and validation controllers.

`prepare_training_v4.py` can rebuild preparation into a new empty directory from
the frozen v4 plain or gzip splits. Selection remains label-blind and class
weights use training labels only. Keep the archived training chat for exact
reproduction of the completed runs.

Saved 9B and 4B comparisons are in `outputs/judge_bench_v4` and
`outputs/judge_bench_v4_qwen35_4b`. Earlier experiment evidence remains archived;
its implementation is available in Git history at commit `7a1ac5d`.
