# Evaluation results

Evaluated-Agent runs are grouped by model:

```text
eval_results/
├── <model>/
│   └── <run_id>.json
└── summary.json
```

Each run file contains one Scenario's checkpoint-level responses and metrics.
`summary.json` is regenerated from every model directory after a successful
evaluation command.
