"""Export the final HALO-8K annotations with their saved response records.

Run from the repository root with ``python3 -m evaluation.export_halo_8k``.
An optional SSH source supplies model runs that are not stored locally. Only
explicitly selected response fields cross that connection or enter the release.
"""

import argparse
import csv
import hashlib
import inspect
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.3"


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def response_key(row):
    return tuple(row[field] for field in ("model", "instance", "scenario", "checkpoint"))


def reasoning_usage(agent):
    """Retain missingness and prefer explicitly reported hosted thinking counts."""
    counts, sources = [], set()
    for call in agent.get("model_calls", []):
        usage = call.get("usage") or {}
        hosted = (usage.get("relay_reported_usage") or {}).get("output_tokens_details") or {}
        thinking = hosted.get("thinking_tokens")
        if isinstance(thinking, int):
            counts.append(thinking)
            sources.add("hosted_thinking_tokens")
        elif usage.get("reasoning_tokens_reported") is not False and isinstance(usage.get("reasoning_tokens"), int):
            counts.append(usage["reasoning_tokens"])
            sources.add("normalized_reasoning_tokens")
        else:
            return None, None
    if not counts:
        return None, None
    return sum(counts), "+".join(sorted(sources))


def selected_responses(results_root, wanted):
    """Read only matching checkpoints and emit an allowlist of release fields."""
    wanted = {tuple(key) for key in wanted}
    models = sorted({key[0] for key in wanted})
    for model in models:
        for path in sorted((Path(results_root) / model).glob("*.json")):
            run = json.loads(path.read_bytes())
            if run.get("model") != model:
                raise ValueError("Run model differs from its directory")
            events = run["scenario"]["events"]
            for point in run["checkpoints"]:
                key = (model, run["instance_id"], run["scenario_id"], point["event_id"])
                if key not in wanted:
                    continue
                agent = point["agent"]
                if agent.get("decision") != "final":
                    raise ValueError(f"Selected checkpoint is not final: {key}")
                prefix = []
                for event_id, event in events.items():
                    prefix.append({"event_id": event_id, **event})
                    if event_id == point["event_id"]:
                        break
                if not prefix or prefix[-1]["event_id"] != point["event_id"]:
                    raise ValueError(f"Checkpoint absent from saved events: {key}")
                trace = agent.get("inspection_trace", [])
                if not all(set(entry) == {"command", "result"} for entry in trace):
                    raise ValueError(f"Unexpected inspection record shape: {key}")
                metrics = point.get("metrics", {})
                reasoning, reasoning_source = reasoning_usage(agent)
                recorded_output = metrics.get("completion_tokens")
                visible_output = recorded_output
                if reasoning_source == "hosted_thinking_tokens":
                    visible_output = None if recorded_output is None else recorded_output - reasoning
                if visible_output is not None and visible_output < 0:
                    raise ValueError(f"Negative visible completion count: {key}")
                yield {
                    "model": model,
                    "domain": run["domain_id"],
                    "need_type": run["trigger_type"],
                    "instance": run["instance_id"],
                    "scenario": run["scenario_id"],
                    "checkpoint": point["event_id"],
                    "instance_description": run["instance"]["instance_description"],
                    "personal_description": run["instance"]["personal_description"],
                    "scenario_description": run["scenario"]["scenario_description"],
                    "event_prefix_json": encode_json(prefix),
                    "inspection_trace_json": encode_json(trace),
                    "final_action_json": encode_json(agent.get("content")),
                    "execution_result_json": encode_json(agent.get("execution_result")),
                    "retrieval_rounds": agent.get("intermediate_round_count", metrics.get("intermediate_round_count")),
                    "input_tokens": metrics.get("input_tokens"),
                    "output_tokens": recorded_output,
                    "completion_tokens": visible_output,
                    "reasoning_tokens": reasoning,
                }


def remote_responses(host, results_root, wanted):
    """Stream selected fields over SSH without downloading raw provider logs."""
    code = "import json\nfrom pathlib import Path\n"
    code += "\n\n".join(inspect.getsource(function) for function in (
        encode_json, reasoning_usage, selected_responses,
    ))
    code += f"\nfor row in selected_responses({str(results_root)!r}, {wanted!r}):\n    print(encode_json(row))\n"
    process = subprocess.Popen(
        ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "python3", "-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        process.stdin.write(code.encode("utf-8"))
        process.stdin.close()
        for line in process.stdout:
            yield json.loads(line)
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"Remote response extraction failed: {error[-2000:]}")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def split_labels(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def validate_release(rows):
    if len(rows) != 8000 or len({row["item_id"] for row in rows}) != 8000:
        raise ValueError("HALO-8K must contain exactly 8,000 unique annotated items")
    model_counts = Counter(row["model"] for row in rows)
    if len(model_counts) != 10 or set(model_counts.values()) != {800}:
        raise ValueError("Expected ten models with 800 annotated responses each")
    scenario_keys = {(row["instance"], row["scenario"]) for row in rows}
    checkpoints = {(row["instance"], row["scenario"], row["checkpoint"]) for row in rows}
    if len(scenario_keys) != 400 or len(checkpoints) != 2160:
        raise ValueError("Unexpected scenario/checkpoint coverage")
    if len({row["instance"] for row in rows}) != 50 or len({row["domain"] for row in rows}) != 10:
        raise ValueError("Unexpected environment/domain coverage")
    per_scenario = Counter((row["model"], row["instance"], row["scenario"]) for row in rows)
    if set(per_scenario.values()) != {2}:
        raise ValueError("Expected two selected checkpoints per model and scenario")
    for row in rows:
        if row["action_level"] not in {"L0", "L1", "L2", "L3"}:
            raise ValueError("Unknown human action level")
        if row["verdict"] not in {"acceptable", "unacceptable"}:
            raise ValueError("Unknown verdict")
        if row["timing"] not in {"Timely", "Premature", "Belated", "Inaction"}:
            raise ValueError("Unknown timing label")
        if row["intensity"] not in {None, "Appropriate", "Insufficient", "Excessive"}:
            raise ValueError("Unknown intensity label")
        if row["intensity"] is None and not (row["action_level"] == "L0" and row["verdict"] == "unacceptable"):
            raise ValueError("Unexpected missing intensity label")
        for field in ("event_prefix_json", "inspection_trace_json", "final_action_json", "execution_result_json"):
            json.loads(row[field])
        text = encode_json(row)
        if re.search(r"(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9_.-]{20,}|/Users/wenqipei/|/data/wenqi/)", text):
            raise ValueError(f"Credential or local-path pattern in item {row['item_id']}")
    return {
        "responses": len(rows),
        "models": dict(sorted(model_counts.items())),
        "domains": len({row["domain"] for row in rows}),
        "environments": len({row["instance"] for row in rows}),
        "scenarios": len(scenario_keys),
        "checkpoints": len(checkpoints),
        "action_levels": dict(sorted(Counter(row["action_level"] for row in rows).items())),
        "verdicts": dict(sorted(Counter(row["verdict"] for row in rows).items())),
        "timing": dict(sorted(Counter(row["timing"] for row in rows).items())),
        "intensity": dict(sorted(Counter(row["intensity"] or "missing" for row in rows).items())),
        "missing_reasoning_counts": sum(row["reasoning_tokens"] is None for row in rows),
    }


def dataset_card(stats, columns, *, overview=False):
    integers = {
        "retrieval_rounds", "input_tokens", "output_tokens",
        "completion_tokens", "reasoning_tokens",
    }
    features = []
    for column in columns:
        if column in {"failed_dimensions", "other_issues"}:
            feature = "list: string"
        else:
            dtype = "int64" if column in integers else "string"
            feature = f"dtype: {dtype}"
        features.append(f"  - name: {column}\n    {feature}")
    features_yaml = "\n".join(features)
    image_block = (
        "![HALO overview](assets/overview.png)\n\n"
        "*Personal environments, agent intervention, and human assessment. "
        "Circled choices are illustrative.*\n\n"
        if overview else ""
    )
    return f'''---
language:
- en
pretty_name: HALO-8K
size_categories:
- 1K<n<10K
tags:
- proactive-agents
- human-evaluation
- tool-use
configs:
- config_name: default
  data_files:
  - split: evaluation
    path: data/halo_8k.jsonl
dataset_info:
  features:
{features_yaml}
---

# ![HALO-8K](assets/halo_wordmark.svg)

<strong><span style="color: #386B9D">H</span>uman-Grounded Gr<span style="color: #C98446">A</span>nular Eva<span style="color: #4B9672">L</span>uation of Pr<span style="color: #8671B1">O</span>active Agents</strong>

[HALO-Arena](https://github.com/CedricPei/HALO/) · [Data format](docs/data_format.md)

HALO-8K is a human-annotated dataset for evaluating when and how proactive
agents should help. Built on HALO-Arena, it captures agents retrieving evidence
from personal environments and deciding how to intervene.

This work was done in collaboration with **2077AI**.

**8,000 responses · 10 models · 2,160 checkpoints · 400 scenarios**

{image_block}## Data

HALO-Arena covers **10 domains**, with **5 distinct simulated users per domain**.
Each user has a separate personal environment, yielding **50 users and
50 environments** in total.

Each record includes observed context, retrieval commands and results, the
final action and execution result, human verdicts and diagnosis, written
rationales, and resource usage. Action levels are final human annotations:
L0 Observation, L1 Attention, L2 Preparation, and L3 Execution.

Each model contributes 800 responses sampled across these environments.

## Quickstart

```python
from datasets import load_dataset
import json

data = load_dataset("CedPei/HALO-8K", split="evaluation")
events = json.loads(data[0]["event_prefix_json"])
retrieval = json.loads(data[0]["inspection_trace_json"])
action = json.loads(data[0]["final_action_json"])
```

`scenario_description` is annotation-side context. Use observed events and
retrieved evidence as agent input. See the [data format](docs/data_format.md)
for field definitions and annotation notes.

<details>
<summary>Citation</summary>

```bibtex
@misc{{pei2026halo,
  title={{HALO: Human-Grounded GrAnular EvaLuation of PrOactive Agents}},
  author={{Pei, Wenqi and Wang, Xinyu and Liu, Jinrui and Zhuge, Cynthia and
          Sheng, Jie and Wang, Zihan and Huang, Tenghao and Chen, Muhao}},
  year={{2026}},
  url={{https://github.com/CedricPei/HALO/}}
}}
```

</details>
'''


def dataset_details(stats):
    model_lines = "\n".join(f"| `{model}` | {count:,} |" for model, count in stats["models"].items())
    return f'''# HALO-8K data format

[Dataset overview](../README.md)

The release has one `evaluation` split. Responses share scenarios and environments
across models and checkpoints, so a random row split does not create independent
scenarios. Each model contributes two selected checkpoints per scenario, rather
than covering every model–checkpoint pair.

## Files and fields

- `data/halo_8k.jsonl`: one row per annotated response.
- `annotations/eval-annotations.csv`: the original 8,000-row annotation export,
  preserving its labels and text exactly.
- `release_manifest.json`: coverage, label counts, and file SHA-256 checksums.

| Field | Meaning |
| --- | --- |
| `item_id` | Stable annotation item ID. |
| `model`, `domain`, `need_type`, `instance`, `scenario`, `checkpoint` | Saved response identity and context grouping. |
| `instance_description`, `personal_description` | Saved environment and user descriptions supplied to the agent. |
| `scenario_description` | Construction-side scenario summary available during annotation. It is **not agent input** and may describe later developments. |
| `event_prefix_json` | Ordered observed events through the selected checkpoint, with event IDs. No later events are included. |
| `inspection_trace_json` | Ordered inspection command/result pairs from that checkpoint. One retrieval round may contain multiple commands. |
| `final_action_json` | Final command, or JSON `null` for no action. |
| `execution_result_json` | Saved result of the final command, or JSON `null`. |
| `action_level` | Human-annotated action level. This is the authoritative evaluation label. |
| `verdict` | `acceptable` or `unacceptable`. |
| `failed_dimensions` | Human-selected rejection dimensions, as a list. |
| `timing` | `Timely`, `Premature`, `Belated`, or `Inaction`. |
| `intensity` | `Appropriate`, `Insufficient`, `Excessive`, or missing. |
| `other_issues` | Optional human-selected issue categories, as a list. |
| `rationale` | Human-written annotation rationale. |
| `retrieval_rounds` | Number of completed intermediate inspection rounds. |
| `input_tokens`, `output_tokens` | Aggregate usage counts across retained calls, including retries. |
| `reasoning_tokens` | Recorded reasoning count. Unreported reasoning is missing. |
| `completion_tokens` | Completion count. For Claude, hosted thinking tokens are subtracted from `output_tokens`. |

The four `*_json` fields contain JSON-encoded strings so heterogeneous app
parameters, query results, and execution results retain their original shapes.
Decode these fields with `json.loads`.

## Annotation semantics

Action levels are L0 (Observation), L1 (Attention), L2 (Preparation), and
L3 (Execution). The `action_level` field contains the final human annotation.

Acceptability is judged at the observed checkpoint. Rejected L0 responses are
assigned Inaction without a further intensity choice. For rejected L1–L3
responses, timing and intensity are separate questions. The export alias
`Timely` denotes appropriate timing. Other issues are Unsupported Evidence,
Irrelevant Action, Factual Contradiction, and Others with a written explanation.

The annotation export contains {stats['verdicts']['acceptable']:,} acceptable
and {stats['verdicts']['unacceptable']:,} unacceptable responses.

| Model | Responses |
| --- | ---: |
{model_lines}

## Context and usage notes

Use the observed event prefix and inspection results to reconstruct the
agent-visible evidence. Keep `scenario_description` out of agent inputs to
avoid revealing construction-side information. The full executable environment
and app definitions are provided in the project repository. This response
release is not a dump of initial environment stores or raw API messages.

For Claude, reasoning counts use the hosted usage field
`output_tokens_details.thinking_tokens`, since its normalized reasoning field
does not carry those counts. Other models use reported normalized reasoning
counts. Missing components remain missing. Tokenization and usage accounting
are provider-dependent. Retrieval rounds count inspection turns, rather than
individual commands or API calls.

'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ROOT / "annotation_app/data/eval-annotations.csv")
    parser.add_argument("--results-root", type=Path, default=ROOT / "eval_results")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/huggingface_halo_8k")
    parser.add_argument("--overview", type=Path, help="Optional overview PNG to include in the dataset card")
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-results-root")
    args = parser.parse_args()
    if bool(args.remote_host) != bool(args.remote_results_root):
        parser.error("Supply both --remote-host and --remote-results-root")
    if args.output.exists():
        parser.error("Output already exists. Choose a new directory to preserve prior exports.")
    if args.overview and not args.overview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        parser.error("--overview must point to a PNG image")
    with args.annotations.open(encoding="utf-8-sig", newline="") as handle:
        annotations = list(csv.DictReader(handle))
    wanted = {response_key(row) for row in annotations}
    if len(wanted) != len(annotations):
        raise ValueError("Duplicate annotated response identities")
    selected = {}

    def accept(record):
        key = response_key(record)
        if key in selected:
            raise ValueError(f"Duplicate saved run for annotated response: {key}")
        selected[key] = record

    for record in selected_responses(args.results_root, wanted):
        accept(record)
    missing = sorted(wanted - selected.keys())
    if missing and args.remote_host:
        for record in remote_responses(args.remote_host, args.remote_results_root, missing):
            accept(record)
    missing = wanted - selected.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} annotated responses in {sorted({key[0] for key in missing})}")
    rows = []
    for annotation in annotations:
        row = {"item_id": annotation["item_id"], **selected[response_key(annotation)]}
        row.update({field: annotation[field] for field in ("action_level", "verdict", "timing", "rationale")})
        row["intensity"] = annotation["intensity"] or None
        row["failed_dimensions"] = split_labels(annotation["failed_dims"])
        row["other_issues"] = split_labels(annotation["other_issues"])
        rows.append(row)
    stats = validate_release(rows)
    (args.output / "data").mkdir(parents=True)
    (args.output / "annotations").mkdir()
    data_path = args.output / "data/halo_8k.jsonl"
    with data_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(encode_json(row) + "\n")
    shutil.copyfile(args.annotations, args.output / "annotations/eval-annotations.csv")
    (args.output / "docs").mkdir()
    (args.output / "docs/data_format.md").write_text(dataset_details(stats), encoding="utf-8")
    (args.output / "assets").mkdir()
    shutil.copyfile(ROOT / "assets/halo_wordmark.svg", args.output / "assets/halo_wordmark.svg")
    if args.overview:
        shutil.copyfile(args.overview, args.output / "assets/overview.png")
    (args.output / "README.md").write_text(dataset_card(stats, rows[0], overview=bool(args.overview)), encoding="utf-8")
    manifest = {
        "dataset": "HALO-8K", "schema_version": SCHEMA_VERSION, "split": "evaluation",
        "statistics": stats,
        "files": {str(path.relative_to(args.output)): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        } for path in sorted(args.output.rglob("*")) if path.is_file()},
    }
    (args.output / "release_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "statistics": stats, "data_bytes": data_path.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
