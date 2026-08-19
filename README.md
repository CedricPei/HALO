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
- `benchmark/` contains the audited manifest and generated benchmark artifacts.

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

For every Instance, two copies of each of the four Trigger types are shuffled
into an eight-Scenario order. Construction uses an Instance Creator, combined
Scenario Creator, Reviewer, deterministic Validator, and final Noise Creator.
A Scenario candidate is committed only after both validation and review pass.
Public Instance metadata lives in `profile.json`; construction outputs live in
`outputs/{instance,review,noise}.json`; accepted and rejected Scenario records
live under `scenarios/logs/`.

All model calls use non-streaming HTTP(S). A construction request receives up
to five immediate network attempts, while Instance, Scenario, Noise, and
malformed Reviewer outputs have separate content/output retry budgets. Five
exhausted network attempts stop the build without treating the transport error
as semantic feedback. Use `--audit-only` to validate existing data and rebuild
`benchmark/manifest.json` without calling a model.

## Evaluation

Evaluation parallelizes independent `(model, Scenario)` jobs. It uses eight
workers by default; checkpoints within each Scenario remain serial.

Every Event prefix is evaluated in a fresh copy of the accepted Instance, so
behavior from an earlier checkpoint cannot affect a later one. Inspection calls
may retrieve information before the Agent returns either no action or one final
App command. Final commands execute once in a disposable state copy. Results,
usage, latency, inspection traces, and intermediate rounds are written beneath
`eval_results/<model>/` and summarized in `eval_results/summary.json`.

```bash
conda run -n proact python -B -m evaluation.runner \
  --instance work_and_professional_life_01 \
  --scenario dependency_1 \
  --workers 8
```

The default evaluated model is `gpt-5.6-sol`. `--model` and `--instance` may be
repeated. Model names beginning with `gemini` use the official Gemini
`generateContent` endpoint; other names use the configured OpenAI-compatible
Responses endpoint.

## Human review

Construction and evaluation review decisions are stored idempotently in
`annotations/reviews.sqlite`. Construction review covers Trigger validity and
causal timing. Evaluation review checks whether an action is timely and has an
appropriate L0–L3 intensity, with structured rejection reasons for timing,
intensity, unsupported evidence, irrelevant action, and factual contradiction.

```bash
conda run -n proact python -B -m streamlit run interfaces/audit.py
conda run -n proact python -B -m streamlit run interfaces/eval.py
```
