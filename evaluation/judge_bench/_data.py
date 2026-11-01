"""Shared preparation helpers for the frozen 7,600/400 split."""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import prompt

DATASET_ID = "judge-bench-v4-sft-7600-validation400-20261013"
LABEL_FIELDS = (
    "verdict", "timing", "intensity", "action_intensity",
    "rejection_dimensions", "other_issue",
)
PLACEHOLDER = "_"


def _json(value: Any, *, sort_keys: bool = True) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"), allow_nan=False,
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _prepare_row(row: dict, split: str) -> dict:
    gold = row["gold"]
    human_comment = gold.get("comments", "")
    if not isinstance(human_comment, str):
        raise ValueError("Human comments must remain strings")
    limit = prompt.JUDGE_SCHEMA["properties"]["rationale"]["maxLength"]
    if not human_comment.strip():
        source = "missing"
    elif len(human_comment) > limit:
        source = "over_schema_limit"
    else:
        source = "human_comments"
    rationale_supervised = source == "human_comments"
    target = {field: gold[field] for field in LABEL_FIELDS}
    target["rationale"] = human_comment if rationale_supervised else PLACEHOLDER
    completion = _json(target, sort_keys=False)
    # Verify, do not repair: raw human labels must obey the existing rubric.
    if prompt.parse_prediction(completion) != target:
        raise ValueError("Prepared target changed a human label")
    loss_prefix_end = len(completion)
    if not rationale_supervised:
        suffix = ',"rationale":"' + PLACEHOLDER + '"}'
        if not completion.endswith(suffix):
            raise ValueError("Unsupervised rationale is not the final JSON member")
        loss_prefix_end = len(completion) - len(PLACEHOLDER) - len('"}')
    messages = prompt.build_messages(row["input_context"])
    messages.append({"role": "assistant", "content": completion})
    return {
        "schema_version": 1, "dataset_id": DATASET_ID,
        "source_benchmark_id": row["benchmark_id"], "item_id": row["item_id"],
        "training_split": split,
        "messages": messages, "completion": completion,
        "supervision": {
            "loss_prefix_end": loss_prefix_end,
            "rationale_supervised": rationale_supervised,
            "mask_completion_suffix": not rationale_supervised,
            "rationale_source": source,
            "placeholder": not rationale_supervised,
            "completion_characters": len(completion),
            "human_comment_characters": len(human_comment),
        },
        # These records are for audit only and must never enter model messages.
        "gold": gold, "provenance": row["provenance"],
        "hashes": {
            "source_input_context_sha256": row["hashes"]["input_sha256"],
            "messages_sha256": _hash(messages),
            "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
        },
    }


def training_distribution(rows: list[dict]) -> dict:
    return {
        "items": len(rows),
        "scenarios": len({row["provenance"]["scenario_key"] for row in rows}),
        "instances": len({row["provenance"]["instance_key"] for row in rows}),
        "by_source_model": _counts(row["provenance"]["source_model"] for row in rows),
        "by_domain": _counts(row["provenance"]["domain_id"] for row in rows),
        "by_trigger": _counts(row["provenance"]["trigger_type"] for row in rows),
        "rationale_sources": _counts(row["supervision"]["rationale_source"] for row in rows),
        "gold": {
            field: _counts("null" if row["gold"][field] is None else row["gold"][field] for row in rows)
            for field in ("verdict", "timing", "intensity", "action_intensity")
        },
    }


def source_distribution(rows: list[dict]) -> dict:
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


def select_validation_ids(rows: list[dict], seed: int, *, holdout_per_scenario: int = 1) -> set[str]:
    if holdout_per_scenario != 1:
        raise ValueError("Validation must contain exactly one item per scenario")
    scenarios, strata, exact = defaultdict(list), defaultdict(list), defaultdict(set)
    for row in rows:
        meta = row["provenance"]
        scenarios[meta["scenario_key"]].append(row)
        exact[row["hashes"]["input_sha256"]].add(row["item_id"])
    models = sorted({row["provenance"]["source_model"] for row in rows})
    if len(rows) != 8000 or len({r["item_id"] for r in rows}) != 8000 or len(scenarios) != 400 or len(models) != 10:
        raise ValueError("Requires the complete original 8000 rows and 400 scenarios")
    for key, group in scenarios.items():
        if len(group) != 20 or Counter(r["provenance"]["source_model"] for r in group) != Counter({m: 2 for m in models}):
            raise ValueError("Each original scenario must have 2 rows per model")
        meta = group[0]["provenance"]
        strata[meta["domain_id"], meta["trigger_type"]].append(key)
    if len(strata) != 40 or any(len(v) != 10 for v in strata.values()):
        raise ValueError("Requires 40 domain/trigger strata with 10 scenarios each")
    rng = random.Random(seed)
    selected = set()
    for stratum, keys in sorted(strata.items()):
        pairs = {}
        for key in sorted(keys):
            candidates = []
            for pair in itertools.combinations(sorted(scenarios[key], key=lambda r: r["item_id"]), holdout_per_scenario):
                ids = {r["item_id"] for r in pair}
                if all(exact[r["hashes"]["input_sha256"]] <= ids for r in pair):
                    candidates.append(pair)
            rng.shuffle(candidates)
            if not candidates:
                raise ValueError("No exact-input-disjoint pair in " + key)
            pairs[key] = candidates
        order = sorted(keys, key=lambda k: (len(pairs[k]), k))
        counts = Counter()
        chosen = {}

        def search(index: int) -> bool:
            if index == len(order):
                return all(counts[m] == holdout_per_scenario for m in models)
            key = order[index]
            for pair in pairs[key]:
                addition = Counter(r["provenance"]["source_model"] for r in pair)
                if any(counts[m] + n > holdout_per_scenario for m, n in addition.items()):
                    continue
                counts.update(addition)
                chosen[key] = pair
                if search(index + 1):
                    return True
                counts.subtract(addition)
                del chosen[key]
            return False

        if not search(0):
            raise ValueError("Cannot satisfy label-blind model quotas in " + str(stratum))
        selected.update(r["item_id"] for pair in chosen.values() for r in pair)
    if len(selected) != 400 * holdout_per_scenario:
        raise ValueError("Holdout selection has the wrong item count")
    return selected


def read_frozen_split(path: Path, expected_sha256: str) -> tuple[list[dict], Path]:
    """Accept a plain or gzip archive while verifying the frozen uncompressed bytes."""
    actual = path if path.is_file() else path.with_suffix(path.suffix + ".gz")
    raw = gzip.decompress(actual.read_bytes()) if actual.suffix == ".gz" else actual.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("A frozen source split changed")
    return [json.loads(line) for line in raw.splitlines() if line.strip()], actual
