# ProActCal

ProActCal is a benchmark for constructing and evaluating proactive-agent
behavior across evolving application state. Each Instance defines one person's
working context, eight Trigger-based Scenarios, and unrelated background Noise.

## Collaborator quick start

If you were asked to help generate benchmark data, first clone the private
repository and create a Python 3.11 environment:

```bash
git clone https://github.com/CedricPei/ProActCal.git
cd ProActCal
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`. Edit `.env` with the Creator model credentials and
endpoint. Never commit `.env`.

Run the Domain and Instance assigned to you:

```bash
python -B build_benchmark.py \
  --domain <domain-id> \
  --instance <instance-id>
```

The target Instance must already be listed by `benchmark_plan.json`. To build
every planned Instance in one Domain, omit `--instance`:

```bash
python -B build_benchmark.py --domain <domain-id>
```

Construction resumes from data already on disk: accepted Scenarios and final
Noise are skipped. One complete Instance contains eight Scenarios and makes
many paid model calls, so confirm the assigned IDs and Creator model before
starting.

### Model configuration

`llm_client.py` reads `.env` from the repository root and uses non-streaming
HTTP(S) requests. Shell environment variables override values in the file.

For a model served through an OpenAI-compatible Responses API, set:

```dotenv
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://your-provider.example
OPENAI_MODEL=<creator-model-id>
OPENAI_REASONING_EFFORT=high
LLM_TIMEOUT=300
```

Do not include `/v1` at the end of `OPENAI_BASE_URL`; the client adds
`/v1/responses`. Set `CREATOR_LLM_PROVIDER=openai` to use this provider.

For the Gemini Developer API, set:

```dotenv
GEMINI_API_KEY=...
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
GEMINI_MODEL=gemini-2.5-flash
GEMINI_THINKING_LEVEL=high
LLM_TIMEOUT=300
```

Set `CREATOR_LLM_PROVIDER=gemini` to use this provider.

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
Before constructing data, update `benchmark_plan.json` and configure the Creator
provider in `.env`.

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
