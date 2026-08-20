# ProActCal

ProActCal is a benchmark for constructing and evaluating proactive-agent
behavior across evolving application state. Each Instance defines one person's
working context, eight Trigger-based Scenarios, and unrelated background Noise.
Evaluation replays every Event prefix in isolation and records the Agent's
inspection and final-action decisions for human review.

## Collaborator quick start

If you were asked to help **run the benchmark**, the normal task is Agent
evaluation, not benchmark construction. First clone the private repository and
create a Python 3.11 environment:

```bash
git clone https://github.com/CedricPei/ProActCal.git
cd ProActCal
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`. Edit `.env` with the credentials and endpoint for
the model being evaluated. Never commit `.env`.

Start with one Scenario as a smoke test:

```bash
python -B -m evaluation.runner \
  --model <model-id> \
  --instance work_and_professional_life_01 \
  --scenario dependency_1 \
  --workers 1
```

After that succeeds, evaluate the same model on every accepted Scenario:

```bash
python -B -m evaluation.runner --model <model-id> --workers 8
```

The full command currently schedules 24 jobs: three accepted Instances times
eight Scenarios. Results are written to `eval_results/<model>/`, and
`eval_results/summary.json` is refreshed when the run finishes without job
failures. Re-running the same command is safe: existing `(model, Scenario)`
results are skipped, so an interrupted run resumes the remaining jobs. A full
evaluation makes many paid model calls; confirm the model ID, credentials, and
smoke-test output before starting it.

### Model configuration

`llm_client.py` reads `.env` from the repository root and uses non-streaming
HTTP(S) requests. Shell environment variables override values in the file.

For a model served through an OpenAI-compatible Responses API, set:

```dotenv
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://your-provider.example
OPENAI_MODEL=<construction-model-id>
OPENAI_REASONING_EFFORT=high
LLM_TIMEOUT=300
```

Do not include `/v1` at the end of `OPENAI_BASE_URL`; the client adds
`/v1/responses`. Evaluation uses this route whenever `--model` does not start
with `gemini`. `OPENAI_MODEL` is required for construction but the evaluation
model is taken directly from `--model`.

For the Gemini Developer API, set:

```dotenv
GEMINI_API_KEY=...
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
GEMINI_MODEL=gemini-2.5-flash
GEMINI_THINKING_LEVEL=high
EVALUATION_GEMINI_TRANSPORT=developer
LLM_TIMEOUT=300
```

Evaluation uses this route whenever `--model` starts with `gemini`.
`GEMINI_MODEL` is required only for construction. To construct benchmark data
with Gemini, also set `CREATOR_LLM_PROVIDER=gemini`; otherwise construction
defaults to the OpenAI-compatible provider.

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

Construction is only for generating or completing benchmark Instances. It is
not required to evaluate the accepted Scenarios already in the repository.
Before constructing data, update `benchmark_plan.json` and configure the
Creator provider in `.env`.

```bash
python -B build_benchmark.py \
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
python -B -m evaluation.runner \
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
python -B build_benchmark.py --audit-only
python -B -m streamlit run interfaces/audit.py
python -B -m streamlit run interfaces/eval.py
```
