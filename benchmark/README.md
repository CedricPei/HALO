# Proactive benchmark

The dataset is organized as `domains/<domain_id>/instances/<instance_id>`.

Each broad Domain is specified by a human in `benchmark_plan.json`. An Instance
contains one protagonist plus empty database and File structures. Its eight
Scenarios add their required facts and File content to one shared initial state,
while every Scenario's Event stream runs independently from it.

The construction plan contains Domain definitions and overall scale, not
Scenario order, IDs, or dates. For every Instance, two copies of each of the four
Trigger types are shuffled into a random eight-Scenario order at runtime. IDs
come from each Trigger's occurrence number (`risk_1`, `risk_2`, and so on), and
dates are chosen randomly but increase with construction order inside each
Instance so accumulated initialization state always predates the next Scenario.
Both IDs and dates remain stable across that Scenario's retries. Construction
processes selected Instances serially. Within each Instance, its eight Scenarios
also run serially because every accepted Scenario updates the shared initial
state and the context used for later date selection and duplicate avoidance.

Construction uses four LLM roles—Instance Creator, combined Scenario Creator,
Reviewer, and one final Noise Creator—plus one deterministic Validator. Each
role receives one static Markdown view from `env/app_surfaces/`: Instance Creator
and Evaluated Agent have dedicated surfaces, while Scenario and Noise creation
share the Scenario surface. These Markdown files are the role-facing command
references; there is no separate command allowlist. Action intensity inference
is private evaluation logic in `evaluation/intensity.py`, and Reviewer receives
no App surface. The Instance
Creator sees existing Instances in its Domain. Scenario Creator and
Reviewer compare compact causal summaries of accepted Scenarios across that
Domain, with same-trigger examples separated for duplicate avoidance. The
Scenario Creator emits Events and optional `ini_data_commands` together. Every
Event has a timezone-aware `time` describing when it enters the observable
environment; these times strictly increase and remain distinct from App record
timestamps and Calendar business times. A candidate is committed only after
Validator and Reviewer pass. After all eight Scenarios, Noise Creator adds one
validated batch of unrelated background data.
Each Instance stores its public description in `profile.json`, its three
construction outputs in `outputs/{instance,review,noise}.json`, and all accepted
and rejected Scenario construction records under one `scenarios/logs/` folder.
Noise is validated before commit without a separate validation artifact. A
build records progress and completion status in `benchmark/build.log`; candidate
success statistics live only in the audited `benchmark/manifest.json`.

Construction and final audit share the `build_benchmark.py` entry point. Use
`--audit-only` to validate existing data and regenerate the Manifest without
calling an LLM. All construction and evaluated-Agent requests share the root
`llm_client.py` HTTP transport and retry implementation.

Every construction LLM request uses non-streaming HTTP(S), a five-minute
per-request timeout, and up to five immediate network attempts. Every network
failure increments the current call's counter regardless of error type; the
fifth failure stops the entire build. Network failures never become semantic
feedback. Instance, Scenario, Noise,
and malformed Reviewer outputs each have a separate budget of up to five
content/output attempts. If all five HTTP attempts fail, the entire build stops
without recording a rejected candidate; it can be resumed after connectivity
recovers. Creator JSON parse failures and malformed Reviewer output are retried
but never written as `failed_*` or included in success statistics. Reports count
only candidates with a semantic outcome: accepted, Validator rejected, or
Reviewer rejected.

## Run a small evaluation

Use the Python 3.11 `proact` environment. Every Event prefix is evaluated in a
fresh copy of the accepted Instance: the Agent sees the prefix but no behavior
from earlier checkpoints. Inspection App calls are private information-gathering
steps with no fixed round limit. Both decisions use one `content` field:
inspection content is an executed list of read commands, while final content is
null or exactly one intended App command. Final Action commands are executed
once in a disposable copy of the checkpoint state; operation timestamps for
sending, drafting, paying, and refunding come from the current Event's `time`.
If the Agent instead places an Inspect-only command in `final`, the response is
retained for human review as L0 (no outward action) and that command is not
executed. Every Action command has an evaluator-side intensity mapping, while
the Agent sees only each command's actual effect and is asked to choose an
effective response. No intensity labels are exposed in its prompt or App
surface.
Database and Files expose retrieval only to the evaluated Agent. Calls use
non-streaming HTTP(S), and results, model-request latency,
intermediate rounds, and inspection traces are saved under
`eval_results/<model>/`. Each model has its own result directory at the project
root.
Evaluated-Agent output receives lightweight JSON cleanup for wrappers and
trailing commas before validation. Invalid output is regenerated at most three
times with the same prompt and no error feedback. If a Final Action cannot be
executed, only that final round is regenerated: all completed inspection
results remain in the prompt, and the Agent gets up to three attempts for the
round. Exhausting the retry budget records the checkpoint as an error and omits
it from human review.

Token reporting separates input, visible completion, hidden reasoning, and
provider-reported total tokens. Tokens and intermediate rounds are retained for
every Prefix–Action checkpoint, then summarized by model with totals and
per-checkpoint averages in `eval_results/summary.json`. Model-call and inspection-
command counts are not reported metrics.

```bash
conda run -n proact python -B -m evaluation.runner \
  --instance work_and_professional_life_01 --scenario dependency_1
```

The default model is `gpt-5.6-sol`. `--model` and `--instance` may both be
repeated. Evaluation runs model/Scenario pairs in parallel, with `--workers 8`
as the default; checkpoints inside one Scenario remain serial. Model names
beginning with `gemini` use the official Gemini `generateContent` HTTP endpoint;
other names use the configured OpenAI-compatible Responses endpoint.

## Human review

Both review workflows write idempotent per-annotator decisions to
`annotations/reviews.sqlite`.

Construction review checks the Scenario-level Trigger and causal trajectory,
including one prefix where intervention remains unacceptably early and one
later prefix where intervention has become unacceptably late. Agent evaluation
uses a binary acceptable/unacceptable label. Acceptable responses imply timely
timing and appropriate intensity; unacceptable responses require a Timing,
Intensity, or Other rejection reason. Intensity is classified as Insufficient or
Excessive. Timing uses Premature, Belated, and Inaction. Other issues use the
two-word categories Unsupported Evidence, Irrelevant Action, and Factual
Contradiction; uncommon issues use Others with free text. The Eval interface
preselects the inferred L0-L3 action intensity so the annotator can confirm or
correct it.

```bash
# Local Streamlit interfaces
conda run -n proact python -B -m streamlit run interfaces/audit.py
conda run -n proact python -B -m streamlit run interfaces/eval.py
```
