"""Freeze label-blind, same-scenario training demonstrations for JudgeBench.

The target checkpoint is excluded across all source models. Selection prefers
four distinct other checkpoints; a shortage can use a second annotated action
at another checkpoint only under the explicitly recorded fallback policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import prompt
from .runner import _atomic_json, _canonical, _hash, _read_jsonl, _rewrite_jsonl


VERSION = "judge-bench-same-scenario-4shot-v1"
INSTRUCTIONS = """

HUMAN-LABELED EXAMPLES
The single background applies to every example and to the target in this scenario.
Four examples show human judgments of other candidate actions. Use these judgments
to calibrate the rubric to the annotators' preferences. Each example supplies its
own checkpoint prefix, retrieval rounds, candidate action, and human judgment.
Examples and target are independent simulations. An event, retrieval result, or
assistant intervention shown only in an example is not an observation at the
target. Assess the target using its own supplied prefix and retrieval results.
Some examples may be different actions at the same other checkpoint. No example
uses the target checkpoint. Human comments are supplied only when available;
missing comments do not imply any missing label. Do not copy a judgment based
only on an action's intensity or on the prevalence of a label in the examples.
Return one JSON judgment for target only, using the original seven-field schema.
"""
POLICY = {
    "example_source": "train.jsonl only; no test annotations are demonstrations",
    "checkpoint_identity": ["provenance.scenario_key", "provenance.event_id"],
    "count": 4,
    "selection": "SHA256 deterministic ordering of checkpoint groups, then rows; maximize distinct other checkpoints first; no gold labels or comments used in selection",
    "fallback": "If only three other checkpoints exist, select a fourth distinct annotated row from one of those checkpoints; never use the target checkpoint",
    "background": "once, shared across examples and target; verify exact equality",
    "duplicate_examples": "four distinct item IDs and input-context hashes",
}
PROMPT_SHA256 = _hash({
    "version": VERSION, "base_prompt_sha256": prompt.PROMPT_SHA256,
    "instructions": INSTRUCTIONS, "policy": POLICY,
    "rendering": "canonical JSON: background, human_labeled_examples, target",
    "labels": ["verdict", "timing", "intensity", "action_intensity", "rejection_dimensions", "other_issue"],
    "optional_human_text": ["comments", "primary_reason", "other_issue_detail"],
})


def select_examples(target: dict, pool: list[dict], seed: int) -> list[dict]:
    """Selection inspects provenance and input hashes, never target/source gold."""
    scenario = target["provenance"]["scenario_key"]
    checkpoint = target["provenance"]["event_id"]
    target_hash = _hash(target["input_context"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        if (row["provenance"]["scenario_key"] == scenario
                and row["provenance"]["event_id"] != checkpoint
                and _hash(row["input_context"]) != target_hash):
            groups[row["provenance"]["event_id"]].append(row)
    key = lambda value: _hash([VERSION, seed, scenario, checkpoint, value])
    for rows in groups.values():
        rows.sort(key=lambda row: key(row["item_id"]))
    chosen: list[dict] = []
    seen: set[str] = set()
    for event in sorted(groups, key=key):
        row = next((row for row in groups[event] if _hash(row["input_context"]) not in seen), None)
        if row is not None:
            chosen.append(row)
            seen.add(_hash(row["input_context"]))
        if len(chosen) == 4:
            break
    if len(chosen) < 4:
        remaining = sorted((row for rows in groups.values() for row in rows), key=lambda row: key(row["item_id"]))
        for row in remaining:
            digest = _hash(row["input_context"])
            if digest not in seen:
                chosen.append(row)
                seen.add(digest)
            if len(chosen) == 4:
                break
    if len(chosen) != 4:
        raise ValueError(f"Fewer than four distinct eligible examples: {target['item_id']}")
    return chosen


def _evidence(row: dict) -> dict:
    return json.loads(prompt.build_messages(row["input_context"])[1]["content"])


def build_messages(target: dict, examples: list[dict]) -> list[dict[str, str]]:
    evidence = _evidence(target)
    background = evidence.pop("background")
    demonstrations = []
    for example in examples:
        demo = _evidence(example)
        if demo.pop("background") != background:
            raise ValueError("Example and target backgrounds differ")
        gold = example["gold"]
        judgment = {field: gold[field] for field in prompt.JUDGE_SCHEMA["required"] if field != "rationale"}
        for field in ("comments", "primary_reason", "other_issue_detail"):
            if isinstance(gold.get(field), str) and gold[field].strip():
                judgment[field] = gold[field]
        demonstrations.append({"checkpoint_id": example["provenance"]["event_id"], **demo, "human_judgment": judgment})
    content = {"background": background, "human_labeled_examples": demonstrations,
               "target": {"checkpoint_id": target["provenance"]["event_id"], **evidence}}
    return [{"role": "system", "content": prompt.SYSTEM_PROMPT + INSTRUCTIONS},
            {"role": "user", "content": _canonical(content).decode("utf-8")}]


def prepare(train: Path, test: Path, output: Path, seed: int = 20261011) -> dict[str, Any]:
    training = _read_jsonl(train)
    targets = _read_jsonl(test)
    training_ids = {row["item_id"] for row in training}
    test_ids = {row["item_id"] for row in targets}
    if len(training_ids) != len(training) or len(test_ids) != len(targets) or training_ids & test_ids:
        raise ValueError("Duplicate or overlapping train/test IDs")
    pool: dict[str, list[dict]] = defaultdict(list)
    for row in training:
        pool[row["provenance"]["scenario_key"]].append(row)
    prepared = []
    checkpoint_counts: Counter[int] = Counter()
    for target in targets:
        examples = select_examples(target, pool[target["provenance"]["scenario_key"]], seed)
        events = [row["provenance"]["event_id"] for row in examples]
        if target["provenance"]["event_id"] in events:
            raise ValueError("Target checkpoint leakage")
        checkpoint_counts[len(set(events))] += 1
        messages = build_messages(target, examples)
        refs = [{"item_id": row["item_id"], "checkpoint_id": row["provenance"]["event_id"],
                 "source_model": row["provenance"]["source_model"],
                 "input_context_sha256": _hash(row["input_context"]),
                 "human_judgment_sha256": _hash(row["gold"])} for row in examples]
        prepared.append({"item_id": target["item_id"], "scenario_key": target["provenance"]["scenario_key"],
                         "target_checkpoint_id": target["provenance"]["event_id"],
                         "input_context_sha256": _hash(target["input_context"]),
                         "messages_sha256": _hash(messages), "examples": refs,
                         "distinct_example_checkpoints": len(set(events)), "messages": messages})
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "prepared_messages.jsonl", output / "example_manifest.json", output / "frozen_prompt.json"]
    if any(path.exists() for path in paths):
        raise ValueError("Prepared artifacts already exist; preserve the frozen selection")
    _rewrite_jsonl(paths[0], prepared)
    manifest = {"version": VERSION, "selection_seed": seed, "policy": POLICY,
                "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
                "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
                "target_count": len(targets), "training_count": len(training),
                "distinct_checkpoint_counts": dict(sorted(checkpoint_counts.items())),
                "prepared_messages_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                "prompt_sha256": PROMPT_SHA256,
                "items": [{key: value for key, value in row.items() if key != "messages"} for row in prepared]}
    _atomic_json(paths[1], manifest)
    _atomic_json(paths[2], {"prompt_version": VERSION, "prompt_sha256": PROMPT_SHA256,
                         "base_prompt_sha256": prompt.PROMPT_SHA256, "schema_sha256": _hash(prompt.JUDGE_SCHEMA),
                         "system_messages": prepared[0]["messages"][:1], "schema": prompt.JUDGE_SCHEMA,
                         "selection_policy": POLICY, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return {key: value for key, value in manifest.items() if key != "items"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    print(json.dumps(prepare(args.train, args.test, args.output_dir, args.seed), ensure_ascii=False))


if __name__ == "__main__":
    main()
