# ProActCal

ProActCal is a benchmark for constructing and evaluating proactive-agent
behavior across evolving application state. Each Instance defines one person's
working context, eight Trigger-based Scenarios, and unrelated background Noise.
Evaluation replays every Event prefix in isolation and records the Agent's
inspection and final-action decisions for human review.

## Repository layout

- `benchmark_plan.json` defines the human-authored Domains and benchmark scale.
- `domains/` contains generated Instances, Scenarios, and their construction
  records.
- `creators/` implements Instance, Scenario, Reviewer, Validator, and Noise
  construction stages.
- `evaluation/` runs evaluated Agents and writes results to `eval_results/`.
- `interfaces/` provides construction-audit and evaluation-review interfaces.
- `benchmark/` contains the audited manifest and detailed workflow
  documentation.

## Construction

Use the Python 3.11 `proact` environment. Model credentials and HTTP(S)
endpoints are read from the local `.env`, which is intentionally excluded from
Git.

```bash
conda run -n proact python -B build_benchmark.py \
  --domain work_and_professional_life \
  --instance work_and_professional_life_01
```

Construction is serial: selected Instances run one at a time, and each
Instance's eight Scenarios run one at a time before final Noise generation.
This preserves the shared initial state, increasing reference dates, and
duplicate-avoidance feedback used by later Scenarios.

## Evaluation

Evaluation parallelizes independent `(model, Scenario)` jobs. It uses eight
workers by default; checkpoints within each Scenario remain serial.

```bash
conda run -n proact python -B -m evaluation.runner \
  --instance work_and_professional_life_01 \
  --scenario dependency_1 \
  --workers 8
```

See [`benchmark/README.md`](benchmark/README.md) for the complete construction,
evaluation, retry, metric, and human-review specifications.
