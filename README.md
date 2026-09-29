# ProActCal

ProActCal evaluates whether an Agent recognizes when help is needed, gathers
relevant evidence, and responds at the right time with an appropriate action.
The active benchmark contains **10 domains, 50 instances, 400 scenarios, and
2,160 event checkpoints per model**.

## Data and repository layout

```text
proactive_agent_benchmark/
├── README.md                  # Project documentation
├── annotations/               # Original human audit: 800 annotation records
├── domains/                   # Revised benchmark used for evaluation
│   └── <domain>/
│       ├── domain.json
│       └── instances/<instance>/
│           ├── profile.json
│           ├── data/          # SQLite database and text files
│           ├── app/           # Shared initial App state, including noise
│           └── scenarios/<trigger>/*.json
├── evaluation/                # Agent runner, prompts and local validation
├── annotation_app/            # Annotation interface
├── env/                       # Simulated Apps and their API surfaces
├── creators/                  # Benchmark construction pipeline
├── benchmark_plan.json        # Domain definitions and planned scale
├── benchmark_catalog.py       # Dataset and evaluation-result loaders
├── build_benchmark.py         # Construction/resume and current-data audit
├── command_parser.py          # Local App command execution
├── llm_client.py               # HTTP(S) model clients
└── requirements.txt
```

Each instance has **one shared initial environment** (`profile.json`, `data/`,
`app/`) for all eight scenarios: two each for `blockage`, `dependency`,
`opportunity`, and `risk`. Scenarios are independent branches. Each evaluation
checkpoint copies that same initial environment and replays only the available
event prefix. Background noise is already materialized in the shared state;
archived construction/noise commands must not be reapplied for evaluation.
Public scenario JSON contains only `scenario_description` and `events`.

The original human audit and adjudication retained 289 scenarios and selected
111 for revision. Revision and subsequent AI acceptance covered all 400;
179 public scenarios and 48 shared instance environments differ from raw.
These are AI acceptance results, not new human votes. The original 800 human
votes and comments are retained in `annotations/proact-anno-audit-annotations.csv`.
The historical raw archive has been removed; `domains/` is the active dataset.

Local credentials and the existing design documents remain in their original
locations. Evaluation creates `eval_results/` when needed; it is not a dataset.

## Initial construction pass rates

These counts describe persisted, parseable candidates from the **initial
construction**, before human audit and benchmark revision.

| Stage | Passed / evaluated | Pass rate | Rejected |
|---|---:|---:|---:|
| Validator | 509 / 568 | 89.61% | 59 |
| Reviewer, after Validator passed | 400 / 509 | 78.59% | 109 |
| Complete pipeline | 400 / 568 | 70.42% | 168 |

Revised candidates count as separate candidates. These are candidate acceptance
rates, not API-request success rates; network retries and unparseable model
outputs are not included. These statistics were recorded before the historical
construction archive was removed.

## Setup

Use Python 3.11 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Keep `.env` local. `llm_client.py` reads it from the repository root; existing
shell variables take precedence. Model calls use HTTP(S).

For the OpenAI-compatible Responses endpoint:

```dotenv
CREATOR_LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_RESPONSES_URL=https://api.openai.com/v1/responses
OPENAI_REASONING_EFFORT=xhigh
```

`OPENAI_RESPONSES_URL` is the complete endpoint; no path is appended. The
construction model configured in code is `gpt-5.6-sol`. Evaluation selects its
model separately with `--model` and does not inherit the construction reasoning
setting.

For the Gemini Developer API:

```dotenv
CREATOR_LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
GEMINI_MODEL=...
GEMINI_THINKING_LEVEL=high
EVALUATION_GEMINI_TRANSPORT=developer
```

## Validate the published dataset

Run local checks without calling a model or writing reports:

```bash
python -B -m evaluation.validate
```

This checks the shared-state structure, database integrity, initial record
timing and wallet consistency, and replays all 2,160 independent event prefixes
across 400 scenarios and 50 databases. Temporary states are removed and all
published source files must remain unchanged. This verifies local execution;
semantic quality and model performance require separate evaluation.

A lighter current-data audit is available:

```bash
python -B build_benchmark.py --audit-only
```

## Agent evaluation

Start with one scenario:

```bash
python -B -m evaluation.runner \
  --model <model-id> \
  --instance work_and_professional_life_01 \
  --scenario dependency_1 \
  --workers 1
```

Then evaluate the complete dataset:

```bash
python -B -m evaluation.runner --model <model-id>
```

Each checkpoint lets the Agent inspect read-only App state and return either
no action or one final command, executed in a disposable state copy. Independent
`(model, scenario)` jobs run in parallel; checkpoints within a scenario are
serial. Both evaluation and checkpoint repair default to **6 workers per process**;
use `--workers` to override this. `--model` and `--instance` may be repeated. Model IDs beginning with
`gemini` use the configured Gemini transport; other IDs use the Responses
endpoint.

Runs are saved under `eval_results/<model>/`, with aggregate metrics in
`eval_results/summary.json`. A successful run is reused only when its input
fingerprint matches the current scenario, shared environment and execution
code and model request settings. Runs retain scenario/profile snapshots for
review; results without these snapshots must be rerun. Old runs are retained
but excluded from current summary metrics when their fingerprints or settings
no longer match.
An explicitly verified scheduling-only revision may be recorded under
`compatible_input_fingerprints`, with its Git revision and reason. The original
fingerprint and checkpoint records remain intact; unrelated code or data
changes still invalidate reuse.

Progress and API retries are logged both to the console and to
`evaluation/evaluation.log`; use `--log-file <path>` for separate model logs.
Logs include checkpoint completion, scenario counts, failures, and a heartbeat
while requests are pending. Normal evaluation automatically gives each failed
checkpoint up to **three additional full attempts**, stopping on success
(at most four full attempts including the initial run). A run still containing
failures exits nonzero. Rerunning the same command repairs only failed checkpoints
in saved scenarios and skips completed scenarios; successful checkpoints are reused.
HTTP 400/401/403/404 errors stop immediately; transient errors use bounded retries.

To retry only failed checkpoints with the same Agent policy and model settings:

```bash
python -u -B -m evaluation.retry --model gemini-3.1-pro-preview --max-attempts 3
```

Each invocation gives every currently failed checkpoint up to three new full
attempts (the default), stopping as soon as it succeeds. Each attempt replays
the original independent prefix. This is separate from the three output/tool
attempts already allowed per Agent round. Successful checkpoints and annotation
IDs stay unchanged. Each attempt atomically updates the existing result file,
preserving all preceding failures under `retry_history`; exhausted failures
remain visible. The retry module does not change the execution fingerprint.

Every model call records input, visible completion, reasoning and total tokens,
plus `latency_ms` for that HTTP response. Checkpoint and scenario metrics sum
their recorded calls, including output/tool retries within the Agent loop.
Latency excludes network-failure backoff, local tool execution and queue time;
it is not wall-clock duration or time to first token. `summary.json` aggregates
the currently selected successful checkpoints, excluding failed checkpoints.
After checkpoint repair, `totals_including_retry_history` additionally counts
the preceding failed attempts; ordinary `totals` describes the selected results.
Keep the first-attempt failures when reporting benchmark reliability separately
from completion after retries.

Export an English summary with per-checkpoint averages (CSV, readable directly in Excel):

```bash
python -B -m evaluation.export
```

This writes only `eval_results/evaluation_summary.csv`.
Only current successful checkpoints contribute; earlier failed attempts are
excluded. `all_success` uses each model's successful points; `common_success`
uses only points successful for every selected model (`--model` is repeatable).
Means weight each checkpoint equally. Latency is the sum of its recorded API
response times in seconds. `retrieval_rounds` counts completed inspect rounds,
not individual tool commands; `api_calls` also counts output retries within
the successful episode.
Token columns separate `input_tokens`, `completion_tokens` (visible output,
excluding reasoning), and `reasoning_tokens`. Each successful checkpoint's
recorded API calls are summed before averaging across checkpoints, without
dataset-wide token consumption totals.

To run both Gemini models concurrently in the background, use separate processes
and moderate concurrency (four scenarios per model):

```bash
mkdir -p eval_results/logs
nohup python -u -B -m evaluation.runner --model gemini-3.8-flash --workers 4 \
  --log-file eval_results/logs/gemini-3.8-flash.log >/dev/null 2>&1 &
nohup python -u -B -m evaluation.runner --model gemini-3.1-pro-preview --workers 4 \
  --log-file eval_results/logs/gemini-3.1-pro-preview.log >/dev/null 2>&1 &
tail -f eval_results/logs/gemini-3.8-flash.log
```

The evaluated Agent can also use `assistant.notify_user(message)` to present
a non-empty, one-way notification to the instance user. The runner fixes the
recipient and simulation timestamp and saves the record as
`agent.execution_result`. This evaluation-only action is unavailable to
construction, Scenario Events and shared initialization. It supplies no user
reply or approval, sends nothing to third parties, and does not perform the
operations described in its text. It still uses the single final-command slot.
Execution verifies the simulated action; it does not prove that a real user
read the message or followed its advice, or that the response is acceptable.

### Action levels and evaluation labels

| Level | Meaning | Examples |
|---|---|---|
| L0 | No action or retrieval | `null`, a retrieval-only final response |
| L1 | Attention | Notify the user, create a reminder, flag an item |
| L2 | Preparation | Prepare a draft, create or update an internal contact |
| L3 | Execution | Send a message, change an active schedule, make a payment |

The interface infers the action level and lets the annotator correct it.
The level describes what the Agent did; **intensity appropriateness** judges
whether that level fits the current need. Higher intensity is not inherently
better. Notifications need evidence and practical value: generic advice,
restating Events, or claiming unperformed operations is insufficient.

The verdict is **Acceptable / Unacceptable**. For an unacceptable non-L0
response, the interface asks:

| Check | Options |
|---|---|
| Timing | Premature / Appropriate / Belated |
| Intensity appropriateness | Insufficient / Appropriate / Excessive |

Timing **Appropriate** is stored as `Timely`. Other issues can also be recorded
as Unsupported Evidence, Irrelevant Action, Factual Contradiction, or Others.
An unacceptable L0 response automatically records **Timing / Inaction**;
Inaction is not a separate selectable timing option. A `null` response hides
the level selector, while a retrieval-only final response retains it.

## Construction and annotation

`python -B build_benchmark.py` constructs or resumes the planned benchmark.
Use `--domain <domain-id>` and optionally `--instance <instance-id>` to limit
scope. Complete published instances are validated and skipped; removing archived
construction logs does not rerun initialization or noise. New construction can
create working outputs and logs for unfinished instances.

The annotation interface lives in `annotation_app/`. The original administrator
uses `ADMIN_USERNAME` and `ADMIN_PASSWORD` from Streamlit Secrets. Additional
accounts can be created or revoked under **Administration → Manage →
Administrator accounts**. Generated passwords are shown once and stored as
salted hashes; revoked accounts lose access on their next page load.
