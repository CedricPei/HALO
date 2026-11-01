"""Matched metrics and prediction checks for the 400-item validation comparison."""

from __future__ import annotations
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any
from .metrics import JOINT_THREE_FIELDS, compute_metrics, joint_label_metrics


MATCH_FIELDS = (
    "input_rows", "prompt_version", "prompt_sha256", "prompt_source_sha256", "schema_sha256",
    "max_tokens", "temperature", "top_p", "presence_penalty", "top_k",
    "thinking", "structured_output", "seed", "concurrency",
    "request_timeout_seconds", "max_retries", "scoring_policy",
)


CORE = (("verdict", "Verdict"), ("timing", "Timing"),
        ("intensity", "Intensity（人工非空）"), ("joint_three", "Joint3（三项全对）"))


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _accuracy_delta_pp(before: dict, after: dict) -> float | None:
    """Compute the percentage-point difference from integer counts, avoiding ties lost to floats."""
    if not before["n"] or not after["n"]:
        return None
    left = Decimal(before["correct"]) / Decimal(before["n"])
    right = Decimal(after["correct"]) / Decimal(after["n"])
    return float((right - left) * Decimal(100))


def _read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _valid_prediction(record: dict | None) -> dict | None:
    if record and record.get("status") == "ok" and isinstance(record.get("prediction"), dict):
        return record["prediction"]
    return None


def analyze(rows: list[dict], predictions: list[dict]) -> dict:
    """Compute label agreement and diagnostic populations without changing labels."""
    metrics = compute_metrics(rows, predictions, bootstrap_repeats=0)
    results = {record["item_id"]: record for record in predictions}
    grouped: dict[str, dict[str, list[dict]]] = {}
    for field in JOINT_THREE_FIELDS:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            value = row["gold"][field]
            groups["__null__" if value is None else str(value)].append(row)
        grouped[field] = groups
    default_pair = [row for row in rows if row["gold"]["timing"] == "Timely" and row["gold"]["intensity"] == "Appropriate"]
    non_default = [row for row in rows if not (row["gold"]["timing"] == "Timely" and row["gold"]["intensity"] == "Appropriate")]
    joint_groups = {
        "by_gold": {field: {label: joint_label_metrics(subset, results, JOINT_THREE_FIELDS)
                             for label, subset in sorted(groups.items())}
                    for field, groups in grouped.items()},
        "default_pair": joint_label_metrics(default_pair, results, JOINT_THREE_FIELDS),
        "non_default_pair": joint_label_metrics(non_default, results, JOINT_THREE_FIELDS),
    }
    errors = {field: Counter() for field in JOINT_THREE_FIELDS}
    for row in rows:
        prediction = _valid_prediction(results.get(row["item_id"]))
        for field in JOINT_THREE_FIELDS:
            gold = row["gold"][field]
            predicted = prediction.get(field) if prediction else "__failed__"
            if prediction is None or field not in prediction or gold != predicted:
                errors[field][("__null__" if gold is None else str(gold),
                               "__null__" if predicted is None else str(predicted))] += 1
    timing_correct = metrics["timing"]["correct"]
    intensity_correct = metrics["intensity"]["correct"]
    # Diagonals avoid deriving integer counts from floating-point recalls.
    def true_positive(field: str, label: str) -> int:
        matrix = metrics[field]["confusion"]
        return matrix["counts"][matrix["rows_gold"].index(label)][matrix["columns_prediction"].index(label)]
    timely = true_positive("timing", "Timely")
    appropriate = true_positive("intensity", "Appropriate")
    joint_correct = metrics["joint_three"]["correct"]
    sources = {
        "timing_correct_from_Timely": {"correct": timely, "total_correct": timing_correct,
                                       "fraction": timely / timing_correct if timing_correct else None},
        "intensity_correct_from_Appropriate": {"correct": appropriate, "total_correct": intensity_correct,
                                                "fraction": appropriate / intensity_correct if intensity_correct else None},
        "joint_three_correct_from_default_pair": {
            "correct": joint_groups["default_pair"]["correct"], "total_correct": joint_correct,
            "fraction": joint_groups["default_pair"]["correct"] / joint_correct if joint_correct else None,
        },
    }
    return {"metrics": metrics, "joint_three_groups": joint_groups, "correct_sources": sources,
            "major_errors": {field: [{"gold": gold, "prediction": prediction, "count": count}
                                     for (gold, prediction), count in counter.most_common()]
                             for field, counter in errors.items()}}


def paired_valid_analysis(rows: list[dict], before_predictions: list[dict], after_predictions: list[dict]) -> dict:
    """Diagnose the same successful-after IDs; never replace full-set headlines."""
    before = {record["item_id"]: record for record in before_predictions}
    after = {record["item_id"]: record for record in after_predictions}
    after_valid = {row["item_id"] for row in rows if _valid_prediction(after.get(row["item_id"])) is not None}
    selected = sorted(item_id for item_id in after_valid if _valid_prediction(before.get(item_id)) is not None)
    selected_set = set(selected)
    subset = [row for row in rows if row["item_id"] in selected_set]
    scores = {}
    for name, predictions in (("before", before_predictions), ("after", after_predictions)):
        metrics = compute_metrics(subset, predictions, bootstrap_repeats=0)
        scores[name] = {key: metrics[key] for key, _ in CORE}
    return {
        "selection_rule": "The same item IDs with structurally valid raw-content predictions after training and before training.",
        "interpretation": "Conditional diagnostic only: selection depends on the after-model output succeeding, so selection bias prevents treating these scores as full-test performance.",
        "items": len(selected), "after_valid_items": len(after_valid),
        "before_invalid_among_after_valid": len(after_valid) - len(selected),
        "excluded_from_full_test": len(rows) - len(selected),
        "item_ids": selected, "item_ids_sha256": _sha(_json(selected)),
        "item_ids_hash_encoding": "UTF-8 JSON array of sorted item IDs; compact separators, ensure_ascii=false",
        "denominators": {key: scores["before"][key]["n"] for key, _ in CORE},
        "before": scores["before"], "after": scores["after"],
        "accuracy_delta_percentage_points": {
            key: _accuracy_delta_pp(scores["before"][key], scores["after"][key])
            for key, _ in CORE
        },
        "primary_metrics_unchanged": True,
    }


def _load_run(directory: Path, rows: list[dict], input_sha256: str) -> tuple[dict, list[dict]]:
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    frozen = json.loads((directory / "frozen_prompt.json").read_text(encoding="utf-8"))
    predictions = _read_jsonl(directory / "predictions.jsonl")
    config = metadata["config"]
    if config.get("input_sha256") != input_sha256 or config.get("input_rows") != len(rows):
        raise ValueError(f"Run input hash/count differs from frozen input: {directory}")
    if metadata.get("complete") is not True or len(predictions) != len(rows):
        raise ValueError(f"Evaluation must finish all {len(rows)} items: {directory}")
    ids = [record["item_id"] for record in predictions]
    if len(set(ids)) != len(ids) or set(ids) != {row["item_id"] for row in rows}:
        raise ValueError(f"Run IDs do not match the frozen input exactly: {directory}")
    if config.get("prompt_sha256") != frozen.get("prompt_sha256") or config.get("schema_sha256") != _sha(_json(frozen["schema"])):
        raise ValueError(f"Frozen prompt/schema and run hashes disagree: {directory}")
    by_id = {record["item_id"]: record for record in predictions}
    for row in rows:
        record = by_id[row["item_id"]]
        if record.get("input_context_sha256") != _sha(_json(row["input_context"])) or not record.get("messages_sha256"):
            raise ValueError(f"Stored item context/message hash is missing or differs: {row['item_id']}")
    files = ("run_metadata.json", "frozen_prompt.json", "predictions.jsonl", "metrics.json")
    provenance = {name: {"sha256": _sha((directory / name).read_bytes()), "path": str((directory / name).resolve())}
                  for name in files if (directory / name).exists()}
    return {"run_directory": str(directory.resolve()), "config": config,
            "frozen_prompt": frozen, "source_artifacts": provenance, **analyze(rows, predictions)}, predictions


def check_match(before: dict, after: dict | None, before_predictions: list[dict], after_predictions: list[dict] | None) -> dict:
    """Model identity may change; evidence, instructions and inference settings may not."""
    if after is None:
        return {"status": "awaiting_after_run", "matched": None, "before_validated": True,
                "required_equal_fields": list(MATCH_FIELDS), "mismatches": [],
                "note": "训练后评测尚未提供；没有训练后分数或变化量。"}
    mismatches = [field for field in MATCH_FIELDS
                  if field not in before["config"] or field not in after["config"]
                  or before["config"][field] != after["config"][field]]
    for field in ("system_messages", "schema"):
        if before["frozen_prompt"].get(field) != after["frozen_prompt"].get(field):
            mismatches.append("frozen_prompt." + field)
    left = {record["item_id"]: record for record in before_predictions}
    right = {record["item_id"]: record for record in (after_predictions or [])}
    for item_id in sorted(left):
        for field in ("input_context_sha256", "messages_sha256"):
            if left[item_id].get(field) != right.get(item_id, {}).get(field):
                mismatches.append(f"{field}:{item_id}")
    return {"status": "matched" if not mismatches else "configuration_mismatch", "matched": not mismatches,
            "required_equal_fields": list(MATCH_FIELDS), "mismatches": mismatches,
            "before_settings": {field: before["config"].get(field) for field in MATCH_FIELDS},
            "after_settings": {field: after["config"].get(field) for field in MATCH_FIELDS},
            "same_item_evidence_and_messages": not any("sha256:" in field for field in mismatches),
            "model_identity_exception": "Model name/weights may change because of training; prompt/evidence/decoder settings must match.",
            "retry_note": "Different reproducible retry seeds may be selected under the saved first-structurally-valid policy; individual stochastic samples are not asserted identical."}
