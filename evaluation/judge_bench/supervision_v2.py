"""Thinking-compatible, class-balanced supervision for the second judge fit.

Human labels and the evaluation prompt stay frozen. The official chat template
supplies an empty reasoning segment; no chain of thought is invented. Every
example teaches the thinking delimiter, JSON structure, and assistant terminator.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from . import prompt
from .prepare_training import LABEL_FIELDS
from ._supervision import _json, _value_spans, encode_completion, label_key, validate_example


PROTOCOL = "thinking-weighted-v2"
FIELD_BUDGETS = {
    "verdict": 2.0, "timing": 2.0, "intensity": 2.0,
    "action_intensity": 1.0, "rejection_dimensions": 1.0, "other_issue": 1.0,
    "rationale": 0.15, "json_structure": 2.0, "assistant_protocol": 2.0,
}
BALANCED_FIELDS = ("verdict", "timing", "intensity", "action_intensity")
CLASS_RATIO_CAP = 3.0


def class_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize capped inverse-square-root weights to training-frequency mean 1."""
    fields = {}
    for field in BALANCED_FIELDS:
        counts = Counter(label_key(row["gold"][field]) for row in rows)
        largest = max(counts.values())
        raw = {key: min(CLASS_RATIO_CAP, math.sqrt(largest / count))
               for key, count in counts.items()}
        mean = sum(counts[key] * raw[key] for key in counts) / len(rows)
        fields[field] = {
            "counts": dict(sorted(counts.items())),
            "weights": {key: raw[key] / mean for key in sorted(raw)},
            "maximum_to_minimum_weight_ratio": max(raw.values()) / min(raw.values()),
        }
    return {
        "method": "sqrt(maximum class count / class count), capped at 3, then normalized to frequency-weighted mean 1",
        "computed_from": "all 7200 training labels only", "ratio_cap": CLASS_RATIO_CAP,
        "fields": fields, "field_loss_budgets": FIELD_BUDGETS,
        "token_length_normalization": "Each field's budget is divided across that field's supervised tokens per example",
        "sampler": "Every training item once per epoch; no minority duplication or dropped majority items",
    }


def encode_example(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    if row.get("schema_version") != 2 or row["supervision"].get("protocol") != PROTOCOL:
        raise ValueError("The weighted thinking encoder requires a version-2 training row")
    messages, completion, policy, target = validate_example(row, FIELD_BUDGETS)

    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    assistant = dict(messages[-1], reasoning_content="")
    official = tokenizer.apply_chat_template(
        [*messages[:-1], assistant], tokenize=False,
        add_generation_prompt=False, enable_thinking=True,
    )
    terminator = "<|im_end|>\n"
    if not official.startswith(prefix) or not official.endswith(terminator):
        raise ValueError("The official assistant template does not extend the evaluation thinking prefix")
    answer = official[len(prefix):-len(terminator)]
    if not answer.endswith(completion):
        raise ValueError("The official template rewrote the target completion")
    closure = answer[:-len(completion)]
    if closure.strip() != "</think>":
        raise ValueError("The official template did not supply an empty, closed reasoning segment")
    return encode_completion(
        row, tokenizer, max_length, target, prefix, answer, closure, FIELD_BUDGETS,
        supervise_thinking_end=True,
    )
