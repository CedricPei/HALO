# HALO

**Human-Grounded Granular Evaluation of Proactive Agents**

HALO evaluates whether an agent recognizes a need for help, retrieves relevant
evidence, and intervenes at an appropriate time and level. It combines an
executable environment with human judgments of proactive assistance.

| Component | Description | Scale |
|---|---|---|
| **HALO-Arena** | Simulated personal apps, SQLite databases, and text files for evidence retrieval and agent actions | 10 domains, 50 personal environments, 400 scenarios, 2,160 event checkpoints |
| **HALO-8K** | Sampled agent responses assessed for action level, acceptability, and reasons for rejection | 8,000 responses from 10 models; 800 responses per model |

The four proactive need types are **Dependency**, **Blockage**, **Risk**, and
**Opportunity**. Each personal environment has two scenarios per need type.
HALO-8K samples two distinct successful checkpoints per model in each scenario;
it does not contain every model–checkpoint combination.

## Repository layout

```text
HALO/
├── domains/                   # HALO-Arena environments and scenarios
│   └── <domain>/
│       ├── domain.json
│       └── instances/<instance>/
│           ├── profile.json
│           ├── data/          # SQLite database and text evidence
│           ├── app/           # Shared initial simulated app state
│           └── scenarios/<trigger>/*.json
├── evaluation/                # Agent runner, validation, export, and sampling
├── eval_results/              # Saved runs and frozen response catalogs
├── annotation_app/            # Streamlit annotation interface and catalogs
├── annotations/               # Benchmark-construction audit annotations
├── env/                       # Simulated apps and command definitions
├── creators/                  # Benchmark construction pipeline
├── benchmark_plan.json        # Domain definitions and planned scale
├── benchmark_catalog.py       # Environment and result loaders
├── build_benchmark.py         # Construction, resume, and dataset audit
├── command_parser.py          # Local command execution
├── llm_client.py              # HTTP(S) model clients
└── requirements.txt
```

Each instance shares one initial environment across its eight independent
scenarios. At every checkpoint, the runner resets that environment and replays
only the observed event prefix. Earlier agent interventions do not carry over
to another checkpoint. Background records are already included in the initial
state and must not be replayed as construction commands.

The response catalogs contain selected actions, inspection traces, execution
results, and model-call records. They are separate from human annotation
exports and from the construction audit in `annotations/`.

## Setup

Use Python 3.11 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Configure the provider and credentials in your local `.env`, following
`.env.example`. Shell environment variables take precedence. Model calls use
HTTP(S) request/response; the client does not require WebSocket transport.

For an OpenAI-compatible provider, set `OPENAI_API_KEY` and
`OPENAI_RESPONSES_URL`. The URL must be the complete Responses endpoint.
For Gemini, set `GEMINI_API_KEY` and the corresponding Gemini settings in the
example file. Evaluation selects its model with `--model`, separately from the
construction model.

Keep real credentials in local environment variables or a private secrets
store. `.env` and `.streamlit/secrets.toml` are excluded from version control;
example configuration files must contain placeholders only.

## Validate HALO-Arena

Run the lighter dataset audit without model calls or generated reports:

```bash
python -B build_benchmark.py --audit-only
```

For full local execution checks:

```bash
python -B -m evaluation.validate
```

The full check validates the shared states and databases and replays all 2,160
independent event prefixes across 400 scenarios. Temporary states are removed,
and source dataset files must remain unchanged. These checks establish local
executability; they do not replace human assessment of response quality.

## Evaluate an agent

Start with one scenario:

```bash
python -B -m evaluation.runner \
  --model <model-id> \
  --instance work_and_professional_life_01 \
  --scenario dependency_1 \
  --workers 1
```

Evaluate the complete arena:

```bash
python -B -m evaluation.runner --model <model-id>
```

The agent may repeatedly inspect read-only evidence, then return no action or
one final command executed in a disposable state copy. All operational apps
are simulated. `assistant.notify_user(message)` records a one-way simulated
notification; it does not supply a user reply or approval.

Independent model–scenario jobs run concurrently, with checkpoints within a
scenario processed in order. The default is six workers; use `--workers` to
adjust it. `--model` and `--instance` may be repeated. Model IDs beginning with
`gemini` use the configured Gemini transport; other IDs use the Responses
endpoint.

Runs are saved under `eval_results/<model>/`, with aggregate metrics in
`eval_results/summary.json`. Successful checkpoints are reused only when the
recorded input and execution settings remain compatible. Rerunning the same
command repairs failed checkpoints and skips completed scenarios.

Normal evaluation gives failed checkpoints up to three additional full
attempts. A run with unresolved failures exits nonzero. To retry only failed
checkpoints under the same model settings:

```bash
python -B -m evaluation.retry --model <model-id> --max-attempts 3
```

Retries preserve earlier failures in `retry_history`. Report initial failure
rates separately from completion after retries. Logs are written to
`evaluation/evaluation.log`; use `--log-file <path>` to choose another location.
Inspect logs and result exports for sensitive content before sharing them.

Export a CSV summary:

```bash
python -B -m evaluation.export
```

The export is saved as `eval_results/evaluation_summary.csv`. `all_success`
uses each model's successful checkpoints; `common_success` uses only
checkpoints successful for every selected model. Resource metrics include
input, visible completion, and reasoning tokens, plus retrieval rounds and
recorded API response time. API response time excludes queueing, local tool
execution, and network-failure backoff.

## HALO-8K and human evaluation

The frozen response catalogs are stored in both `eval_results/` and
`annotation_app/data/`:

- `human_eval_4000.json.gz`: 4,000 responses from five models.
- `human_eval_4000_batch2.json.gz`: 4,000 responses from five additional models.

Together they contain 8,000 distinct responses from 10 models. Each model
contributes 800 responses across all 400 scenarios. Sampling balances event
positions without filtering on action choice or response quality, using seed
`20261002`. The catalogs preserve source-file hashes and original item IDs;
new evaluation runs do not change the frozen annotation tasks.

To reproduce the catalogs when all corresponding source runs are available:

```bash
python -B -m evaluation.human_subset
python -B -m evaluation.human_subset --batch2
```

These commands refuse to overwrite a different published release. The
catalogs are included independently of whether every original source run is
available in this checkout. Human acceptability results require the separate
annotation exports; runner summaries alone are not human quality scores.

### Annotation labels

| Action level | Meaning |
|---|---|
| **L0 — Observation** | Observe or retrieve information without initiating an intervention |
| **L1 — Attention** | Draw attention to relevant information or a need |
| **L2 — Preparation** | Prepare an intervention or organize internal information without an external commitment |
| **L3 — Execution** | Carry out an intervention that changes active commitments or creates external effects |

Annotators confirm or correct the inferred action level and assign an
**Acceptable / Unacceptable** verdict. Higher action levels do not imply
better assistance; restraint can be acceptable when intervention is unwarranted.

Rejected L0 responses are labeled **Inaction**. Rejected L1–L3 responses receive
separate timing judgments (**Premature / Appropriate / Belated**) and intensity
judgments (**Insufficient / Appropriate / Excessive**). Optional other issues
are **Unsupported Evidence**, **Irrelevant Action**, **Factual Contradiction**,
and **Others**, with a written explanation for Others.

### Run the annotation interface

Configure `DATABASE_URL`, `ADMIN_USERNAME`, and `ADMIN_PASSWORD` privately using
environment variables or Streamlit Secrets, following
`annotation_app/secrets.example.toml`. Then run:

```bash
streamlit run annotation_app/app.py
```

Use a separate database and private credentials for your own deployment.
Participant access codes, administrator credentials, account assignments, and
identifying annotation exports must be handled privately. Preserve frozen
catalog IDs and saved annotations when updating an existing deployment.

## Construct new environments

```bash
python -B build_benchmark.py --domain <domain-id> --instance <instance-id>
```

Construction requires configured model access. Complete published instances
are validated and skipped. Use `--audit-only` to inspect the existing arena;
construction is not required to run evaluation on it.
