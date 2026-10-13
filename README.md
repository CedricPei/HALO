# HALO

**Human-Grounded Granular Evaluation of Proactive Agents**

HALO evaluates when and how proactive agents should intervene, using executable
personal environments and human judgments of their responses.

- **HALO-Arena:** 10 domains, 50 environments, 400 scenarios, and 2,160 checkpoints.
- **HALO-8K:** 8,000 responses from 10 models, labeled for action level,
  acceptability, and rejection reasons.

Environments are in [`domains/`](domains/). Frozen response catalogs are in
[`annotation_app/data/`](annotation_app/data/).
Judge evaluation and LoRA training tools are in
[`evaluation/judge_bench/`](evaluation/judge_bench/README.md).

## Setup

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set model credentials and endpoints in `.env` using [`.env.example`](.env.example).

## Usage

```bash
# Validate the dataset without model calls
python -B build_benchmark.py --audit-only

# Evaluate a model across all checkpoints
python -B -m evaluation.runner --model <model-id>

# Export evaluation summaries
python -B -m evaluation.export
```

Runs and summaries are saved in `eval_results/`. Each checkpoint starts from
an independent copy of the simulated environment.

To run human annotation, configure `DATABASE_URL`, `ADMIN_USERNAME`, and
`ADMIN_PASSWORD` using
[`annotation_app/secrets.example.toml`](annotation_app/secrets.example.toml), then run:

```bash
streamlit run annotation_app/app.py
```

Keep credentials and participant access codes private.
