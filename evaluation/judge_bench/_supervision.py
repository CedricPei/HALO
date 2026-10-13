"""Shared row preparation and token supervision for versioned judge fits."""

from __future__ import annotations

import json
from array import array
from typing import Any

from . import prompt
from .prepare_training import LABEL_FIELDS, _prepare_row


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def label_key(value: Any) -> str:
    return "__null__" if value is None else str(value)


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


def prepare_weighted_row(
    source: dict, split: str, *, schema_version: int, dataset_id: str,
    protocol: str, enable_thinking: bool, field_budgets: dict, balance: dict,
) -> dict:
    row = _prepare_row(source, split)
    previous = row["supervision"]
    row.update(schema_version=schema_version, dataset_id=dataset_id, input_context=source["input_context"])
    row["supervision"] = {
        "protocol": protocol, "chat_template_enable_thinking": enable_thinking,
        "rationale_supervised": previous["rationale_supervised"],
        "rationale_source": previous["rationale_source"],
        "placeholder": previous["placeholder"],
        "human_comment_characters": previous["human_comment_characters"],
        "mask_rationale_value": not previous["rationale_supervised"],
        "json_suffix_supervised": True, "assistant_terminator_supervised": True,
        "thinking_end_supervised": enable_thinking, "field_loss_budgets": field_budgets,
        "class_loss_weights": {field: info["weights"][label_key(source["gold"][field])]
                               for field, info in balance["fields"].items()},
    }
    return row


def validate_example(row: dict, field_budgets: dict) -> tuple:
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
    if policy["field_loss_budgets"] != field_budgets:
        raise ValueError("Training loss budgets changed")
    has_rationale = policy["rationale_supervised"]
    if has_rationale:
        if target["rationale"] != row["gold"]["comments"]:
            raise ValueError("A supervised human rationale was rewritten")
    elif target["rationale"] != "_":
        raise ValueError("Absent rationale must use the masked schema placeholder")

    return messages, completion, policy, target


def encode_completion(
    row: dict, tokenizer: Any, max_length: int, target: dict,
    prefix: str, answer: str, closure: str, field_budgets: dict,
    *, supervise_thinking_end: bool,
) -> dict:
    policy = row["supervision"]
    completion = row["completion"]
    has_rationale = policy["rationale_supervised"]
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
    groups: dict[str, list[int]] = {key: [] for key in field_budgets}
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
    if supervise_thinking_end:
        if not any(ids[index] == think_end_id for index in groups["assistant_protocol"]):
            raise ValueError("The thinking end token must receive supervision")
    elif any(ids[index] == think_end_id for index in groups["assistant_protocol"]):
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
        budget = field_budgets[field] * policy["class_loss_weights"].get(field, 1.0)
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
