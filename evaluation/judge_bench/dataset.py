"""Build a label-blind scenario holdout from the two frozen human Eval cohorts.

Only ``input_context`` is suitable for model input. Gold labels and provenance
are deliberately separate. The test set holds out one complete scenario in each
domain/trigger stratum; shared user instances are reported in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_ID = "judge-bench-v1-20261010"
SCHEMA_VERSION = 1
SEED = 20261010
CATALOG_PATHS = (
    ROOT_DIR / "annotation_app/data/human_eval_4000.json.gz",
    ROOT_DIR / "annotation_app/data/human_eval_4000_batch2.json.gz",
)
ANNOTATIONS_PATH = ROOT_DIR / "annotations/eval-annotations.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs/judge_bench_v1"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _normalize_inspection(command: dict) -> dict:
    # The original runner normalizes these surface fields before execution.
    return {
        "app": command["app"].strip().casefold(),
        "function": command["function"].strip().casefold(),
        "parameter": command.get("parameter") or {},
    }


def retrieval_rounds(agent: dict) -> list[dict]:
    """Recover executed round boundaries without retaining model text/thinking."""
    trace = agent["inspection_trace"]
    round_commands: dict[int, list] = {}
    for call in agent["model_calls"]:
        parsed = call.get("parsed")
        if (
            isinstance(parsed, dict)
            and str(parsed.get("decision", "")).strip().casefold() == "inspect"
            and not call.get("attempt_error")
            and not call.get("protocol_error")
        ):
            commands = parsed["content"]
            round_commands[call["round"]] = (
                commands if isinstance(commands, list) else [commands]
            )
    rounds, offset = [], 0
    for index in range(agent["intermediate_round_count"]):
        if index not in round_commands:
            raise ValueError(f"Missing successful inspection round {index}")
        commands = round_commands[index]
        results = trace[offset : offset + len(commands)]
        if [_normalize_inspection(command) for command in commands] != [
            result["command"] for result in results
        ]:
            raise ValueError(f"Inspection commands do not match trace in round {index}")
        rounds.append({
            "round": index + 1,
            "results": [
                {"command": result["command"], "result": result["result"]}
                for result in results
            ],
        })
        offset += len(commands)
    if offset != len(trace):
        raise ValueError("Inspection trace contains unmatched results")
    return rounds


def input_context(item: dict) -> dict:
    """The complete allowlist of evidence visible to the judge."""
    return {
        "background": {
            **{
                key: item["instance"][key]
                for key in ("instance_description", "personal_description")
            },
            "scenario_description": item["scenario_description"],
        },
        "checkpoint_prefix": item["event_prefix"],
        "retrieval_rounds": retrieval_rounds(item["agent"]),
        "candidate_action": item["agent"]["content"],
    }


def _gold(annotation: dict[str, str]) -> dict:
    dimensions = [
        value.strip()
        for value in annotation["diagnostics.rejection_dimensions"].split("|")
        if value.strip()
    ]
    gold = {
        "verdict": annotation["verdict"],
        "timing": annotation["timing"],
        "intensity": annotation["intensity"] or None,
        "action_intensity": annotation["diagnostics.action_intensity"],
        "primary_reason": annotation["primary_reason"],
        "rejection_dimensions": dimensions,
        "other_issue": annotation["diagnostics.other_issue"] or None,
        "other_issue_detail": annotation["diagnostics.other_issue_detail"] or None,
        "comments": annotation["comments"],
        "human_annotation": dict(annotation),
    }
    enums = {
        "verdict": {"acceptable", "unacceptable"},
        "timing": {"Timely", "Premature", "Belated", "Inaction"},
        "intensity": {"Appropriate", "Excessive", "Insufficient", None},
        "action_intensity": {"L0", "L1", "L2", "L3"},
        "other_issue": {
            "Unsupported Evidence", "Irrelevant Action", "Factual Contradiction",
            "Others", None,
        },
    }
    for field, allowed in enums.items():
        if gold[field] not in allowed:
            raise ValueError(f"Unexpected human {field}: {gold[field]!r}")
    if set(dimensions) - {"Timing", "Intensity", "Other issues"}:
        raise ValueError("Unexpected human rejection dimension")
    if gold["intensity"] is None and not (
        gold["verdict"] == "unacceptable" and gold["action_intensity"] == "L0"
    ):
        raise ValueError("Missing intensity outside an unacceptable L0 annotation")
    return gold


def _select_scenarios(items: list[dict], seed: int) -> list[str]:
    """Select from scenario metadata only, before reading human annotations."""
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    scenarios: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        key = item["scenario_key"]
        scenarios[key].append(item)
        groups[item["domain_id"], item["trigger_type"]].add(key)
    models = {item["model"] for item in items}
    if (
        len(items) != 8000 or len({item["item_id"] for item in items}) != 8000
        or len(scenarios) != 400 or len(groups) != 40 or len(models) != 10
        or len({item["domain_id"] for item in items}) != 10
        or len({item["trigger_type"] for item in items}) != 4
    ):
        raise ValueError("Frozen cohorts do not match the expected 8000-item design")
    for key, members in scenarios.items():
        if len(members) != 20 or Counter(item["model"] for item in members) != Counter(
            {model: 2 for model in models}
        ):
            raise ValueError(f"Scenario is not balanced across ten models: {key}")
    rng = random.Random(seed)
    selected = []
    for stratum, keys in sorted(groups.items()):
        if len(keys) != 10:
            raise ValueError(f"Stratum does not contain ten scenarios: {stratum}")
        selected.append(rng.choice(sorted(keys)))
    return sorted(selected)


def _distribution(rows: list[dict]) -> dict:
    label = lambda value: "null" if value is None else str(value)
    return {
        "items": len(rows),
        "scenarios": len({row["provenance"]["scenario_key"] for row in rows}),
        "instances": len({row["provenance"]["instance_key"] for row in rows}),
        "by_source_model": _counts(row["provenance"]["source_model"] for row in rows),
        "by_domain": _counts(row["provenance"]["domain_id"] for row in rows),
        "by_trigger": _counts(row["provenance"]["trigger_type"] for row in rows),
        "by_domain_trigger": _counts(
            f"{row['provenance']['domain_id']}/{row['provenance']['trigger_type']}"
            for row in rows
        ),
        "gold": {
            field: _counts(label(row["gold"][field]) for row in rows)
            for field in ("verdict", "timing", "intensity", "action_intensity", "other_issue")
        },
        "rejection_dimensions": _counts(
            dimension for row in rows for dimension in row["gold"]["rejection_dimensions"]
        ),
        "retrieval_rounds": _counts(str(len(row["input_context"]["retrieval_rounds"])) for row in rows),
        "input_utf8_bytes": {
            "minimum": min(len(_json(row["input_context"]).encode("utf-8")) for row in rows),
            "maximum": max(len(_json(row["input_context"]).encode("utf-8")) for row in rows),
        },
    }


def _hash_overlap(train: list[dict], test: list[dict], field: str) -> dict:
    train_counts = Counter(row["hashes"][field] for row in train)
    test_counts = Counter(row["hashes"][field] for row in test)
    common = train_counts.keys() & test_counts.keys()
    return {
        "shared_unique_hashes": len(common),
        "train_rows_with_shared_hash": sum(train_counts[key] for key in common),
        "test_rows_with_shared_hash": sum(test_counts[key] for key in common),
        "within_train_duplicate_rows": len(train) - len(train_counts),
        "within_test_duplicate_rows": len(test) - len(test_counts),
    }


def _overlap(train: list[dict], test: list[dict]) -> dict:
    result = {}
    for field in ("scenario_key", "instance_key"):
        train_keys = {row["provenance"][field] for row in train}
        test_keys = {row["provenance"][field] for row in test}
        shared = sorted(train_keys & test_keys)
        result[field] = {
            "train_unique": len(train_keys), "test_unique": len(test_keys),
            "shared_count": len(shared), "shared_keys": shared,
        }
    for field in ("input_sha256", "context_sha256", "candidate_action_sha256"):
        result[field] = _hash_overlap(train, test, field)
    result["nonnull_candidate_action_sha256"] = _hash_overlap(
        [row for row in train if row["input_context"]["candidate_action"] is not None],
        [row for row in test if row["input_context"]["candidate_action"] is not None],
        "candidate_action_sha256",
    )
    result["interpretation"] = (
        "Scenario holdout: no scenario/checkpoint occurs in both splits. User instances "
        "and their reusable background/stored records may occur in both splits. This is "
        "not an instance holdout. Repeated generic/no-op actions are expected; action "
        "overlap alone is not identical judge input. Exact full-input and context "
        "overlap are reported independently."
    )
    if result["scenario_key"]["shared_count"]:
        raise ValueError("Scenario leakage across train/test")
    return result


def _duplicate_annotation_audit(rows: list[dict]) -> dict:
    """Expose conflicting human labels on exact inputs without editing labels."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["hashes"]["input_sha256"]].append(row)
    fields = (
        "verdict", "timing", "intensity", "action_intensity",
        "rejection_dimensions", "other_issue",
    )
    conflicts, minimum_verdict_errors = [], 0
    duplicate_groups = 0
    for digest, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        duplicate_groups += 1
        labels = [{key: row["gold"][key] for key in fields} for row in group]
        verdict_counts = Counter(row["gold"]["verdict"] for row in group)
        minimum_verdict_errors += len(group) - max(verdict_counts.values())
        if len({_json(label) for label in labels}) > 1:
            conflicts.append({
                "input_sha256": digest,
                "items": [
                    {"item_id": row["item_id"], "gold": label}
                    for row, label in zip(group, labels)
                ],
            })
    return {
        "exact_input_duplicate_groups": duplicate_groups,
        "conflicting_gold_groups": len(conflicts),
        "conflicts": conflicts,
        "minimum_verdict_errors_for_consistent_predictions": minimum_verdict_errors,
        "maximum_verdict_accuracy_for_consistent_predictions": (
            len(rows) - minimum_verdict_errors
        ) / len(rows),
        "policy": "Original human labels retained; no adjudication or deduplication",
    }


def build_dataset(
    catalog_paths: tuple[Path, ...] = CATALOG_PATHS,
    annotations_path: Path = ANNOTATIONS_PATH,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], dict]:
    items, origins, sources = [], {}, []
    for path in catalog_paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            catalog = json.load(handle)
        sources.append({
            "path": _relative(path), "sha256": _file_hash(path),
            "subset_id": catalog["subset_id"], "items": len(catalog["items"]),
        })
        for item in catalog["items"]:
            if item["item_id"] in origins:
                raise ValueError(f"Duplicate catalog item: {item['item_id']}")
            origins[item["item_id"]] = {
                "catalog": _relative(path), "subset_id": catalog["subset_id"],
            }
            items.append(item)
    selected = _select_scenarios(items, seed)
    test_scenarios = set(selected)
    # This order is intentional: no gold data is loaded during scenario selection.
    annotations = {}
    with annotations_path.open(encoding="utf-8-sig", newline="") as handle:
        for annotation in csv.DictReader(handle):
            item_id = annotation["item_id"]
            if item_id in annotations:
                raise ValueError(f"Multiple human annotations for item: {item_id}")
            annotations[item_id] = annotation
    if annotations.keys() != origins.keys():
        raise ValueError("Annotations must match the frozen 8000-item catalog exactly")
    sources.append({
        "path": _relative(annotations_path), "sha256": _file_hash(annotations_path),
        "items": len(annotations),
    })
    train, test = [], []
    for item in sorted(items, key=lambda entry: (
        entry["scenario_key"], entry["event_id"], entry["model"], entry["item_id"],
    )):
        evidence = input_context(item)
        split = "test" if item["scenario_key"] in test_scenarios else "train"
        provenance = {
            **origins[item["item_id"]],
            **{key: item[key] for key in (
                "source_path", "scenario_key", "event_id", "source_result",
                "domain_id", "trigger_type",
            )},
            "source_model": item["model"],
            "instance_key": item["source_path"].split("/scenarios/", 1)[0],
            "source_action_intensity": item["agent"]["action_intensity"],
        }
        row = {
            "schema_version": SCHEMA_VERSION, "benchmark_id": BENCHMARK_ID,
            "item_id": item["item_id"], "split": split,
            "provenance": provenance, "input_context": evidence,
            "gold": _gold(annotations[item["item_id"]]),
            "hashes": {
                "input_sha256": _hash(evidence),
                "context_sha256": _hash({
                    key: value for key, value in evidence.items() if key != "candidate_action"
                }),
                "candidate_action_sha256": _hash(evidence["candidate_action"]),
            },
        }
        (test if split == "test" else train).append(row)
    for split, rows, per_model, per_domain, per_trigger in (
        ("test", test, 80, 80, 200), ("train", train, 720, 720, 1800),
    ):
        if len(rows) != per_model * 10:
            raise ValueError(f"Unexpected {split} size")
        for field, expected in (
            ("source_model", per_model), ("domain_id", per_domain),
            ("trigger_type", per_trigger),
        ):
            if any(count != expected for count in Counter(
                row["provenance"][field] for row in rows
            ).values()):
                raise ValueError(f"Unexpected {split} {field} distribution")
    manifest = {
        "schema_version": SCHEMA_VERSION, "benchmark_id": BENCHMARK_ID,
        "seed": seed, "sources": sources,
        "selection": {
            "method": "One complete scenario chosen uniformly per domain/trigger stratum",
            "algorithm": "random.Random(seed).choice(sorted(scenarios)); strata sorted lexically",
            "uses_human_labels": False, "uses_judge_predictions": False,
            "strata": 40, "scenarios_per_stratum": 10, "test_scenarios": selected,
            "test_scenarios_sha256": _hash(selected), "items_per_scenario": 20,
        },
        "context_policy": {
            "model_input_key": "input_context",
            "allowed_fields": [
                "background", "checkpoint_prefix", "retrieval_rounds", "candidate_action",
            ],
            "background_fields": [
                "instance_description", "personal_description", "scenario_description",
            ],
            "human_view_alignment": (
                "Include the full scenario description shown to human annotators, "
                "including objective, emerging need, decision context, and consequence. "
                "This follows the explicit user request to provide available human-view context."
            ),
            "retrieval_round_numbering": "1-based; executed command/result pairs only",
            "excluded": [
                "future_events", "model_thinking", "model_calls",
                "final_action_execution_result", "human_labels", "human_comments",
                "source_model", "item_id", "provenance",
            ],
            "truncation": "none; all available prefix/retrieval evidence preserved",
        },
        "label_policy": {
            "source": "Original human CSV, with the full unmodified row retained under gold.human_annotation",
            "empty_diagnostic_values": "Converted to JSON null in normalized gold; original empty strings retained",
            "intensity_null": "Unacceptable L0 rows have no intervention-intensity error; do not impute Appropriate",
            "action_intensity": "Human-selected label, including corrections to automatic intensity",
            "adjudication": "No relabelling, imputation, consensus, or filtering by label",
        },
        "distributions": {"test": _distribution(test), "train": _distribution(train)},
        "overlap": _overlap(train, test),
        "duplicate_annotation_audit": {
            "test": _duplicate_annotation_audit(test),
            "train": _duplicate_annotation_audit(train),
        },
    }
    return test, train, manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(_json(row) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    test, train, manifest = build_dataset()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("test", test), ("train", train)):
        path = args.output_dir / f"{name}.jsonl"
        _write_jsonl(path, rows)
        manifest.setdefault("outputs", {})[name] = {
            "file": path.name, "items": len(rows), "sha256": _file_hash(path),
        }
    path = args.output_dir / "split_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({
        "benchmark_id": BENCHMARK_ID, "test": len(test), "train": len(train),
        "test_scenarios": 40, "train_scenarios": 360,
        "shared_instances": manifest["overlap"]["instance_key"]["shared_count"],
        "shared_full_inputs": manifest["overlap"]["input_sha256"]["shared_unique_hashes"],
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
