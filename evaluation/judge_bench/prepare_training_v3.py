"""Prepare all 7200 training items spanning the full 400 benchmark scenarios.

No validation split is made and no test input is read. Original human labels,
comments, context, and prompt are preserved in separately versioned artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import prompt
from ._supervision import prepare_weighted_row
from .prepare_training import LABEL_FIELDS, _distribution, _file_hash, _hash, _json
from .supervision_v3 import FIELD_BUDGETS, PROTOCOL, class_balance


DATASET_ID = "judge-bench-v3-sft-all7200-20261011"


def prepare(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenarios: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scenarios[row["provenance"]["scenario_key"]].append(row)
    models = {row["provenance"]["source_model"] for row in rows}
    if (len(rows) != 7200 or len({row["item_id"] for row in rows}) != 7200
            or any(row.get("split") != "train" for row in rows)
            or len(scenarios) != 400 or len(models) != 10
            or any(len(group) != 18 for group in scenarios.values())
            or Counter(r["provenance"]["source_model"] for r in rows) != Counter({model: 720 for model in models})):
        raise ValueError("Input must be the new 7200-row, all-400-scenario training split")
    balance = class_balance(rows)
    prepared = []
    for source in rows:
        row = prepare_weighted_row(
            source, "train", schema_version=3, dataset_id=DATASET_ID,
            protocol=PROTOCOL, enable_thinking=False,
            field_budgets=FIELD_BUDGETS, balance=balance,
        )
        prepared.append(row)
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        source_groups[row["hashes"]["source_input_context_sha256"]].append(row)
    conflicts = [group for group in source_groups.values()
                 if len({_hash({field: row["gold"][field] for field in LABEL_FIELDS}) for row in group}) > 1]
    return prepared, {
        "schema_version": 3, "dataset_id": DATASET_ID, "rows": 7200, "test_set_used": False,
        "training_protocol": PROTOCOL,
        "prompt": {"version": prompt.PROMPT_VERSION, "sha256": prompt.PROMPT_SHA256,
                   "source_sha256": _file_hash(Path(prompt.__file__)),
                   "context_policy": "Original frozen build_messages(input_context), unchanged full evidence"},
        "internal_validation": {"enabled": False, "items": 0, "scenarios": 0,
                                "selection": "None; all 7200 examples train, final checkpoint is evaluated once"},
        "class_balance": balance,
        "supervision_contract": {
            "label_fields": list(LABEL_FIELDS), "completion_order": [*LABEL_FIELDS, "rationale"],
            "generation_prefix": "Official tokenizer template with enable_thinking=false, matching evaluation",
            "reasoning": "No reasoning supervision; the official already-closed thinking prefix is masked",
            "prompt_loss": "All system/user and generation-prefix tokens masked",
            "human_rationale": "Original nonblank comment up to 1000 characters; field loss budget 0.1",
            "missing_or_overlong_rationale": "Mask only placeholder string contents and crossing tokens; JSON suffix and assistant end always supervised",
            "truncate": False,
        },
        "human_label_audit": {
            "six_label_structure_valid": 7200, "six_label_consistent_with_rubric": 7200,
            "relabelled": 0, "dropped": 0,
            "rationale_sources": dict(Counter(row["supervision"]["rationale_source"] for row in prepared)),
            "conflicting_exact_input_groups": len(conflicts),
            "conflicting_group_item_ids": [[r["item_id"] for r in group] for group in conflicts],
        },
        "distributions": {"all": _distribution(prepared), "train": _distribution(prepared),
                          "validation": _distribution([])},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/judge_bench_v3/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/judge_bench_v3/training"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new empty output directory; existing training artifacts are immutable")
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    prepared, manifest = prepare(rows)
    manifest["source_train"] = {"path": str(args.input.resolve()), "sha256": _file_hash(args.input), "items": 7200}
    manifest["preparation_source_sha256"] = _file_hash(Path(__file__))
    manifest["supervision_source_sha256"] = _file_hash(Path(__file__).with_name("supervision_v3.py"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chat = args.output_dir / "chat.jsonl"
    with chat.open("w", encoding="utf-8", newline="\n") as handle:
        for row in prepared:
            handle.write(_json(row) + "\n")
    manifest["output"] = {"file": chat.name, "items": 7200, "sha256": _file_hash(chat)}
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset_id": DATASET_ID, "fit": 7200, "validation": 0,
                      "class_balance": balance_summary(manifest), "output_dir": str(args.output_dir.resolve())}))


def balance_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {field: info["weights"] for field, info in manifest["class_balance"]["fields"].items()}


if __name__ == "__main__":
    main()
