"""Dependency-free, failure-aware metrics for frozen JudgeBench runs."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any


VERDICTS = ("acceptable", "unacceptable")
TIMINGS = ("Timely", "Premature", "Belated", "Inaction")
INTENSITIES = ("Appropriate", "Excessive", "Insufficient")
ACTION_INTENSITIES = ("L0", "L1", "L2", "L3")
REJECTION_DIMENSIONS = ("Timing", "Intensity", "Other issues")
OTHER_ISSUES = (
    "Unsupported Evidence", "Irrelevant Action", "Factual Contradiction", "Others"
)
FAILED = "__failed__"
NULL_LABEL = "__null__"
JOINT_THREE_FIELDS = ("verdict", "timing", "intensity")
JOINT_SIX_FIELDS = (*JOINT_THREE_FIELDS, "action_intensity", "rejection_dimensions", "other_issue")


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def classification_metrics(
    pairs: list[tuple[str, str]], labels: tuple[str, ...]
) -> dict[str, Any]:
    """Include invalid outputs as an additional prediction column, never drop them."""
    labels = tuple(labels)
    columns = labels + tuple(sorted({p for _, p in pairs} - set(labels)))
    counts = Counter(pairs)
    support = Counter(g for g, _ in pairs)
    predicted = Counter(p for _, p in pairs)
    n = len(pairs)
    correct = sum(counts[label, label] for label in labels)
    per_class: dict[str, Any] = {}
    for label in labels:
        tp = counts[label, label]
        fp = predicted[label] - tp
        fn = support[label] - tp
        per_class[label] = {
            "support": support[label], "predicted": predicted[label],
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    active = [label for label in labels if support[label]]
    expected = sum(support[label] * predicted[label] for label in labels) / n**2 if n else 0
    accuracy = _ratio(correct, n)
    return {
        "n": n, "correct": correct, "accuracy": accuracy,
        "macro_f1": sum(per_class[l]["f1"] for l in labels) / len(labels) if n else None,
        "balanced_accuracy": (
            sum(per_class[l]["recall"] for l in active) / len(active) if active else None
        ),
        "cohen_kappa": (
            (accuracy - expected) / (1 - expected) if n and expected < 1 else None
        ),
        "confusion": {
            "rows_gold": list(labels), "columns_prediction": list(columns),
            "counts": [[counts[g, p] for p in columns] for g in labels],
        },
        "per_class": per_class,
    }


def _prediction(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result and result.get("status") == "ok" and isinstance(result.get("prediction"), dict):
        return result["prediction"]
    return None


def _pairs(
    rows: list[dict[str, Any]], results: dict[str, dict[str, Any]], field: str,
    *, nonnull: bool = False, null_label: bool = False, valid_only: bool = False,
) -> list[tuple[str, str]]:
    pairs = []
    for row in rows:
        gold = row["gold"].get(field)
        if nonnull and gold is None:
            continue
        pred = _prediction(results.get(row["item_id"]))
        if valid_only and pred is None:
            continue
        predicted = pred.get(field) if pred else FAILED
        if null_label:
            gold = NULL_LABEL if gold is None else gold
            predicted = NULL_LABEL if pred and predicted is None else predicted
        if predicted is None:
            predicted = NULL_LABEL if pred else FAILED
        pairs.append((gold, predicted))
    return pairs


def multilabel_metrics(
    rows: list[dict[str, Any]], results: dict[str, dict[str, Any]], *, valid_only: bool = False
) -> dict[str, Any]:
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    support: Counter[str] = Counter()
    exact = 0
    failures = 0
    n = 0
    sample_f1_sum = 0.0
    for row in rows:
        pred = _prediction(results.get(row["item_id"]))
        if valid_only and pred is None:
            continue
        n += 1
        gold = set(row["gold"]["rejection_dimensions"])
        predicted = set(pred["rejection_dimensions"]) if pred else set()
        failures += pred is None
        exact += pred is not None and gold == predicted
        support.update(gold)
        tp.update(gold & predicted)
        fp.update(predicted - gold)
        fn.update(gold - predicted)
        denom = len(gold) + len(predicted)
        if pred is not None:
            sample_f1_sum += 2 * len(gold & predicted) / denom if denom else 1.0
    per_class = {}
    for label in REJECTION_DIMENSIONS:
        denom = 2 * tp[label] + fp[label] + fn[label]
        per_class[label] = {
            "support": support[label], "tp": tp[label], "fp": fp[label], "fn": fn[label],
            "precision": _ratio(tp[label], tp[label] + fp[label]),
            "recall": _ratio(tp[label], tp[label] + fn[label]),
            "f1": 2 * tp[label] / denom if denom else 0.0,
        }
    sum_tp, sum_fp, sum_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    denom = 2 * sum_tp + sum_fp + sum_fn
    return {
        "n": n, "failed_predictions": failures, "exact_match_accuracy": _ratio(exact, n),
        "micro_f1": 2 * sum_tp / denom if denom else (float(failures == 0) if n else None),
        "macro_f1": sum(x["f1"] for x in per_class.values()) / len(REJECTION_DIMENSIONS) if n else None,
        "samples_f1": _ratio(sample_f1_sum, n), "per_class": per_class,
        "failure_handling": "Invalid predictions have empty predicted labels; exact-match and sample F1 count them as zero.",
    }


def joint_label_metrics(
    rows: list[dict[str, Any]], results: dict[str, dict[str, Any]], fields: tuple[str, ...],
) -> dict[str, Any]:
    """Require exact human-label agreement, independently of rubric consistency.

    JSON null is an ordinary matching value. A missing field is never equivalent
    to null, and a failed/absent response cannot match. Rejection dimensions are
    multilabel sets, so their order does not affect semantic label agreement.
    """
    correct = failed = null_correct = 0
    field_correct: Counter[str] = Counter()
    null_rows = 0
    for row in rows:
        pred = _prediction(results.get(row["item_id"]))
        gold = row["gold"]
        null_rows += gold.get("intensity") is None
        failed += pred is None
        matches = []
        for field in fields:
            matched = pred is not None and field in pred
            if matched and field == "rejection_dimensions":
                matched = isinstance(pred[field], list) and set(gold[field]) == set(pred[field])
            elif matched:
                matched = gold[field] == pred[field]
            matches.append(matched)
            field_correct[field] += matched
        matched = all(matches)
        correct += matched
        null_correct += matched and gold.get("intensity") is None
    return {
        "n": len(rows), "correct": correct, "accuracy": _ratio(correct, len(rows)),
        "fields": list(fields), "failed_predictions": failed,
        "per_field_correct": {field: field_correct[field] for field in fields},
        "gold_null_intensity_rows": null_rows,
        "correct_on_gold_null_intensity": null_correct,
        "diagnostic_consistency_gating": False,
        "policy": "All rows; every selected field must match its human label. Null equals null, missing is not null. Rejection dimensions compare as sets. Internal consistency is separate.",
    }


def _majority_baseline(
    pairs: list[tuple[str, str]], labels: tuple[str, ...],
) -> dict[str, Any]:
    support = Counter(gold for gold, _ in pairs)
    majority = max(labels, key=lambda label: support[label]) if pairs else None
    report = classification_metrics([(gold, majority) for gold, _ in pairs], labels)
    report["predicted_constant_label"] = majority
    report["policy"] = "Always predict the most frequent human class in this scoring population; ties use the declared label order. Analytic reference only, no inference or prompt change."
    return report


def _prediction_bias(
    pairs: list[tuple[str, str]], labels: tuple[str, ...],
) -> dict[str, Any]:
    gold = Counter(value for value, _ in pairs)
    predicted = Counter(value for _, value in pairs)
    n = len(pairs)
    valid = n - predicted[FAILED]
    majority = max(labels, key=lambda label: gold[label]) if pairs else None
    return {
        "n": n, "valid_predictions": valid, "failed_predictions": predicted[FAILED],
        "gold_counts": dict(sorted(gold.items())),
        "prediction_counts": dict(sorted(predicted.items())),
        "gold_fractions": {label: _ratio(count, n) for label, count in sorted(gold.items())},
        "prediction_fractions_all_rows": {
            label: _ratio(count, n) for label, count in sorted(predicted.items())
        },
        "prediction_fractions_valid_rows": {
            label: _ratio(count, valid) for label, count in sorted(predicted.items()) if label != FAILED
        },
        "gold_majority_label": majority,
        "gold_majority_fraction": _ratio(gold[majority], n) if majority is not None else None,
        "predicted_majority_fraction_all_rows": _ratio(predicted[majority], n) if majority is not None else None,
        "predicted_majority_fraction_valid_rows": _ratio(predicted[majority], valid) if majority is not None else None,
        "denominator_policy": "Gold and all-row prediction fractions use this metric's full population. Valid-row prediction fractions exclude failed/absent responses only; a structurally valid null remains a prediction.",
    }


def _majority_baselines(rows: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    verdict = _majority_baseline(_pairs(rows, results, "verdict"), VERDICTS)
    timing = _majority_baseline(_pairs(rows, results, "timing"), TIMINGS)
    intensity = _majority_baseline(_pairs(rows, results, "intensity", nonnull=True), INTENSITIES)
    intensity_null = _majority_baseline(
        _pairs(rows, results, "intensity", null_label=True), (*INTENSITIES, NULL_LABEL)
    )
    labels = {
        "verdict": verdict["predicted_constant_label"],
        "timing": timing["predicted_constant_label"],
        "intensity": intensity_null["predicted_constant_label"],
    }
    prediction = {key: None if value == NULL_LABEL else value for key, value in labels.items()}
    constant_results = {
        row["item_id"]: {"status": "ok", "prediction": prediction}
        for row in rows
    }
    joint = joint_label_metrics(rows, constant_results, JOINT_THREE_FIELDS)
    joint["predicted_constant_labels"] = prediction
    joint["policy"] = "Constant marginal-majority verdict/timing/intensity labels; intensity majority uses all rows, including null. This is an analytic gold-distribution reference, not a fitted or schema-complete model response."
    return {
        "verdict": verdict, "timing": timing, "intensity": intensity,
        "intensity_including_null": intensity_null, "joint_three": joint,
    }


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def grouped_bootstrap(
    rows: list[dict[str, Any]], results: dict[str, dict[str, Any]], *, repeats: int, seed: int
) -> dict[str, Any]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        pred = _prediction(results.get(row["item_id"]))
        group = row.get("provenance", {}).get("scenario_key", row["item_id"])
        groups[str(group)].append((row["gold"]["verdict"], pred["verdict"] if pred else FAILED))
    keys = sorted(groups)
    if not keys or repeats <= 0:
        return {"repeats": 0, "scenario_groups": len(keys)}
    rng = random.Random(seed)
    names = ("accuracy", "macro_f1", "balanced_accuracy", "cohen_kappa")
    samples: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(repeats):
        pairs = [pair for key in rng.choices(keys, k=len(keys)) for pair in groups[key]]
        score = classification_metrics(pairs, VERDICTS)
        for name in names:
            if score[name] is not None:
                samples[name].append(score[name])
    return {
        "method": "percentile cluster bootstrap; resample complete scenario groups with replacement",
        "confidence_level": 0.95, "repeats": repeats, "seed": seed,
        "scenario_groups": len(keys),
        "intervals": {name: [_quantile(vals, 0.025), _quantile(vals, 0.975)] for name, vals in samples.items()},
    }


def compute_metrics(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]], *, bootstrap_repeats: int = 2000,
    seed: int = 20261010,
) -> dict[str, Any]:
    results = {result["item_id"]: result for result in predictions}
    statuses = Counter(results.get(row["item_id"], {}).get("status", "missing") for row in rows)
    valid = statuses.get("ok", 0)
    valid_results = [results[row["item_id"]] for row in rows if _prediction(results.get(row["item_id"]))]
    consistent = sum(record.get("diagnostic_consistency") is True for record in valid_results)
    inconsistent = sum(record.get("diagnostic_consistency") is False for record in valid_results)
    consistency_errors = Counter(
        record.get("diagnostic_consistency_error", "unspecified consistency error")
        for record in valid_results if record.get("diagnostic_consistency") is False
    )
    strict_results = {
        key: (record if record.get("diagnostic_consistency") is True else {**record, "status": "consistency_error"})
        for key, record in results.items()
    }
    rejected = [row for row in rows if row["gold"]["verdict"] == "unacceptable"]
    non_timely = [row for row in rows if row["gold"]["timing"] != "Timely"]
    non_appropriate = [
        row for row in rows if row["gold"]["intensity"] not in ("Appropriate", None)
    ]
    null_intensity = [row for row in rows if row["gold"]["intensity"] is None]
    report = {
        "total": len(rows), "valid_predictions": valid, "failed_predictions": len(rows) - valid,
        "status_counts": dict(sorted(statuses.items())),
        "verdict": classification_metrics(_pairs(rows, results, "verdict"), VERDICTS),
        "verdict_valid_only": classification_metrics(_pairs(rows, results, "verdict", valid_only=True), VERDICTS),
        "verdict_with_strict_consistency": classification_metrics(_pairs(rows, strict_results, "verdict"), VERDICTS),
        "joint_three": joint_label_metrics(rows, results, JOINT_THREE_FIELDS),
        "joint_six": joint_label_metrics(rows, results, JOINT_SIX_FIELDS),
        "diagnostic_consistency": {
            "structurally_valid_predictions": valid, "checked": consistent + inconsistent,
            "consistent": consistent, "inconsistent": inconsistent,
            "not_checked": valid - consistent - inconsistent,
            "consistency_rate_checked": _ratio(consistent, consistent + inconsistent),
            "consistent_fraction_all_rows": _ratio(consistent, len(rows)),
            "error_counts": dict(sorted(consistency_errors.items())),
        },
        "timing": classification_metrics(_pairs(rows, results, "timing"), TIMINGS),
        "timing_non_timely": classification_metrics(
            _pairs(non_timely, results, "timing"), tuple(label for label in TIMINGS if label != "Timely")
        ),
        "intensity": classification_metrics(_pairs(rows, results, "intensity", nonnull=True), INTENSITIES),
        "intensity_non_appropriate": classification_metrics(
            _pairs(non_appropriate, results, "intensity"), tuple(label for label in INTENSITIES if label != "Appropriate")
        ),
        "intensity_null_cases": classification_metrics(
            _pairs(null_intensity, results, "intensity", null_label=True), (NULL_LABEL,)
        ),
        "intensity_including_null": classification_metrics(
            _pairs(rows, results, "intensity", null_label=True), (*INTENSITIES, NULL_LABEL)
        ),
        "action_intensity": classification_metrics(_pairs(rows, results, "action_intensity"), ACTION_INTENSITIES),
        "other_issue_including_null": classification_metrics(
            _pairs(rows, results, "other_issue", null_label=True), (*OTHER_ISSUES, NULL_LABEL)
        ),
        "other_issue_when_gold_present": classification_metrics(
            _pairs(rows, results, "other_issue", nonnull=True), OTHER_ISSUES
        ),
        "rejection_dimensions": multilabel_metrics(rows, results),
        "rejection_dimensions_on_rejected": multilabel_metrics(rejected, results),
        "rejection_dimensions_valid_only": multilabel_metrics(rows, results, valid_only=True),
        "metric_policy": {
            "main_denominator": "All frozen benchmark rows; malformed JSON/schema or absent predictions count as incorrect. Structurally valid verdicts are scored even when auxiliary labels are internally inconsistent.",
            "diagnostic_consistency": "Cross-field rubric errors are reported separately; raw labels are never corrected. The strict-consistency verdict metric requires both a correct verdict and a successful consistency check.",
            "joint_three": "Verdict, timing, and intensity must all equal their human labels on the same row; null intensity remains in the full denominator. No prediction-consistency gate.",
            "joint_six": "All six human labels must match; rejection dimensions are compared as sets. Rationale prose and internal consistency are not gates.",
            "intensity_denominator": "Only rows with a non-null human intensity label; null cases reported separately.",
            "non_default_subsets": "Timing excludes gold Timely; intensity includes only gold Excessive/Insufficient, excluding Appropriate and inapplicable null. Each subset's macro F1 uses its predefined non-default classes; wrong default/null predictions remain confusion columns and count as incorrect.",
            "macro_f1": "Unweighted F1 over all predefined labels, including labels with zero support (zero F1).",
            "balanced_accuracy": "Unweighted recall over gold labels present in the evaluated subset.",
        },
        "baselines": {"always_majority": _majority_baselines(rows, results)},
        "prediction_bias": {
            "verdict": _prediction_bias(_pairs(rows, results, "verdict"), VERDICTS),
            "timing": _prediction_bias(_pairs(rows, results, "timing"), TIMINGS),
            "intensity": _prediction_bias(_pairs(rows, results, "intensity", nonnull=True), INTENSITIES),
            "intensity_including_null": _prediction_bias(
                _pairs(rows, results, "intensity", null_label=True), (*INTENSITIES, NULL_LABEL)
            ),
        },
        "breakdown": {},
    }
    for field in ("domain_id", "trigger_type", "source_model", "action_intensity"):
        by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row["gold"][field] if field == "action_intensity" else row.get("provenance", {}).get(field, "unknown")
            by_value[str(value)].append(row)
        report["breakdown"][field] = {
            value: classification_metrics(_pairs(subset, results, "verdict"), VERDICTS)
            for value, subset in sorted(by_value.items())
        }
    report["verdict_confidence_intervals"] = grouped_bootstrap(
        rows, results, repeats=bootstrap_repeats, seed=seed
    )
    return report
