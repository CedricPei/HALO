"""Non-thinking, class-balanced supervision for the shared-scenario judge fit.

Human labels and the evaluation prompt stay frozen. The official chat template
supplies the same closed thinking prefix used in evaluation. Prefix loss is
masked; label values, JSON structure, and assistant termination are supervised.
"""

from __future__ import annotations

import json
import math
from array import array
from collections import Counter
from typing import Any

from . import prompt
from .prepare_training import LABEL_FIELDS


PROTOCOL = "nonthinking-weighted-v3"
FIELD_BUDGETS = {
    "verdict": 3.0, "timing": 3.0, "intensity": 3.0,
    "action_intensity": 1.0, "rejection_dimensions": 1.0, "other_issue": 1.0,
    "rationale": 0.1, "json_structure": 1.0, "assistant_protocol": 1.0,
}
BALANCED_FIELDS = ("verdict", "timing", "intensity", "action_intensity")
CLASS_RATIO_CAP = 5.0


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def label_key(value: Any) -> str:
    return "__null__" if value is None else str(value)


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
        "method": "sqrt(maximum class count / class count), capped at 5, then normalized to frequency-weighted mean 1",
        "computed_from": f"all {len(rows)} training labels only", "ratio_cap": CLASS_RATIO_CAP,
        "fields": fields, "field_loss_budgets": FIELD_BUDGETS,
        "token_length_normalization": "Each field's budget is divided across that field's supervised tokens per example",
        "sampler": "Every training item once per epoch; no minority duplication or dropped majority items",
    }


def _value_spans(target: dict[str, Any], completion: str) -> dict[str, tuple[int, int]]:
    pieces, spans, cursor = [], {}, 1
    for field in (*LABEL_FIELDS, "rationale"):
        head = _json(field) + ":"
        value = _json(target[field])
        pieces.append(head + value)
        spans[field] = (cursor + len(head), cursor + len(head) + len(value))
        cursor += len(head) + len(value) + 1
    if "{" + ",".join(pieces) + "}" != completion:
        raise ValueError("Completion is not in the versioned canonical field order")
    return spans


def encode_example(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    if row.get("schema_version") != 3 or row["supervision"].get("protocol") != PROTOCOL:
        raise ValueError("The non-thinking encoder requires a version-3 training row")
    messages, completion, policy = row["messages"], row["completion"], row["supervision"]
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Training messages must have exactly system/user/assistant roles")
    if messages[:-1] != prompt.build_messages(row["input_context"]):
        raise ValueError("Training messages differ from the frozen evidence-only prompt")
    if messages[-1]["content"] != completion:
        raise ValueError("Assistant message differs from the completion")
    target = prompt.parse_prediction(completion)
    if any(target[field] != row["gold"][field] for field in LABEL_FIELDS):
        raise ValueError("A training label differs from the original human annotation")
    if policy["field_loss_budgets"] != FIELD_BUDGETS:
        raise ValueError("Training loss budgets changed")
    has_rationale = policy["rationale_supervised"]
    if has_rationale:
        if target["rationale"] != row["gold"]["comments"]:
            raise ValueError("A supervised human rationale was rewritten")
    elif target["rationale"] != "_":
        raise ValueError("Absent rationale must use the masked schema placeholder")

    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assistant = dict(messages[-1], reasoning_content="")
    official = tokenizer.apply_chat_template(
        [*messages[:-1], assistant], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    terminator = "<|im_end|>\n"
    if not official.startswith(prefix) or not official.endswith(terminator):
        raise ValueError("The official assistant template does not extend the non-thinking evaluation prefix")
    answer = official[len(prefix):-len(terminator)]
    if not answer.endswith(completion):
        raise ValueError("The official template rewrote the target completion")
    closure = answer[:-len(completion)]
    if closure != "" or not prefix.endswith("<think>\n\n</think>\n\n"):
        raise ValueError("Official non-thinking generation prefix differs from the evaluation contract")
    start = len(prefix) + len(closure)
    end = start + len(completion)
    spans = {field: (start + left, start + right)
             for field, (left, right) in _value_spans(target, completion).items()}
    # Mask the actual string contents, including escapes, rather than its quotes
    # or the rest of the JSON object. A boundary token touching a placeholder is
    # also masked, so it cannot teach invented rationale text.
    rationale_left, rationale_right = spans["rationale"]
    rationale_content = (rationale_left + 1, rationale_right - 1)
    encoded = tokenizer(prefix + answer, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    groups: dict[str, list[int]] = {key: [] for key in FIELD_BUDGETS}
    labels = [-100] * len(ids)
    for index, (left, right) in enumerate(encoded["offset_mapping"]):
        if right <= left or left < len(prefix) or right > end:
            continue
        if left < start:
            if right <= start:
                groups["assistant_protocol"].append(index)
                labels[index] = ids[index]
            continue
        if left < rationale_content[1] and right > rationale_content[0]:
            if has_rationale:
                groups["rationale"].append(index)
                labels[index] = ids[index]
            continue
        group = next((field for field in LABEL_FIELDS
                      if left < spans[field][1] and right > spans[field][0]), "json_structure")
        groups[group].append(index)
        labels[index] = ids[index]
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if any(value is None or value == tokenizer.unk_token_id for value in (end_id, think_end_id)):
        raise ValueError("Official Qwen tokenizer lacks the required protocol tokens")
    ids.append(end_id)
    labels.append(end_id)
    groups["assistant_protocol"].append(len(ids) - 1)
    if any(ids[index] == think_end_id for index in groups["assistant_protocol"]):
        raise ValueError("Thinking delimiters belong to the masked non-thinking prefix")
    if not groups["json_structure"] or any(not groups[field] for field in LABEL_FIELDS):
        raise ValueError("A target field or JSON structure has no supervised tokens")
    if not has_rationale and groups["rationale"]:
        raise ValueError("A placeholder rationale accidentally receives supervision")
    if len(ids) > max_length:
        raise ValueError(f"{row['item_id']}: {len(ids)} tokens exceeds {max_length}; evidence was not truncated")
    weights = [0.0] * len(ids)
    for field, positions in groups.items():
        if not positions:
            continue
        budget = FIELD_BUDGETS[field] * policy["class_loss_weights"].get(field, 1.0)
        for position in positions:
            weights[position] = budget / len(positions)
    if any((label == -100) != (weight == 0.0) for label, weight in zip(labels, weights)):
        raise ValueError("Loss weights and label masks disagree")
    return {
        "input_ids": array("I", ids), "labels": array("i", labels),
        "loss_weights": array("f", weights), "length": len(ids),
        "supervised_tokens": sum(label != -100 for label in labels), "item_id": row["item_id"],
        "supervised_tokens_by_field": {field: len(value) for field, value in groups.items()},
        "official_thinking_closure": closure, "think_end_token_id": think_end_id,
        "assistant_end_token_id": end_id,
    }
