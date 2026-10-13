"""Prepare 7,600 fit / 400 validation judgments from the original 8,000 labels.

Hold out one complete input per known scenario, balancing source models within
each domain/trigger stratum without consulting labels. Exact duplicate inputs
stay together. Validation labels never determine training class weights.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from . import prompt
from .dataset import _distribution as source_distribution, _hash_overlap
from .prepare_training import LABEL_FIELDS, _distribution, _file_hash, _hash, _json, _prepare_row
from .resplit import select_test_ids
from .supervision_v3 import FIELD_BUDGETS, PROTOCOL, class_balance, label_key


DATASET_ID = "judge-bench-v4-sft-7600-validation400-20261013"


def prepare(rows: list[dict], seed: int) -> tuple[dict[str, list[dict]], list[dict], dict]:
    selected = select_test_ids(rows, seed, holdout_per_scenario=1)
    partitions: dict[str, list[dict]] = {"train": [], "validation": []}
    for source in sorted(rows, key=lambda row: row["item_id"]):
        row = deepcopy(source)
        split = "validation" if row["item_id"] in selected else "train"
        row["source_partition"] = {"benchmark_id": source["benchmark_id"], "split": source["split"]}
        row.update(benchmark_id=DATASET_ID, split=split)
        if _hash(row["input_context"]) != row["hashes"]["input_sha256"]:
            raise ValueError("The original full input changed")
        partitions[split].append(row)
    for name, per_scenario, per_model in (("train", 19, 760), ("validation", 1, 40)):
        members = partitions[name]
        counts = Counter(row["provenance"]["scenario_key"] for row in members)
        models = Counter(row["provenance"]["source_model"] for row in members)
        if len(counts) != 400 or set(counts.values()) != {per_scenario} or set(models.values()) != {per_model}:
            raise ValueError("Scenario or source-model quotas changed")
    overlap = {field: _hash_overlap(partitions["train"], partitions["validation"], field)
               for field in ("input_sha256", "context_sha256", "candidate_action_sha256")}
    if overlap["input_sha256"]["shared_unique_hashes"]:
        raise ValueError("An exact complete input crosses the validation boundary")
    balance = class_balance(partitions["train"])
    prepared = []
    for split, members in partitions.items():
        for source in members:
            row = _prepare_row(source, split)
            previous = row["supervision"]
            row.update(schema_version=3, dataset_id=DATASET_ID, input_context=source["input_context"])
            row["supervision"] = {
                "protocol": PROTOCOL, "chat_template_enable_thinking": False,
                "rationale_supervised": previous["rationale_supervised"],
                "rationale_source": previous["rationale_source"],
                "placeholder": previous["placeholder"],
                "human_comment_characters": previous["human_comment_characters"],
                "mask_rationale_value": not previous["rationale_supervised"],
                "json_suffix_supervised": True, "assistant_terminator_supervised": True,
                "thinking_end_supervised": False, "field_loss_budgets": FIELD_BUDGETS,
                "class_loss_weights": {field: info["weights"][label_key(source["gold"][field])]
                                       for field, info in balance["fields"].items()},
            }
            prepared.append(row)
    manifest = {
        "schema_version": 3, "dataset_id": DATASET_ID, "seed": seed,
        "rows": len(prepared), "test_set_used": False, "independent_test_set": None,
        "training_protocol": PROTOCOL, "class_balance": balance,
        "prompt": {"version": prompt.PROMPT_VERSION, "sha256": prompt.PROMPT_SHA256,
                   "source_sha256": _file_hash(Path(prompt.__file__)),
                   "context_policy": "Original frozen full evidence and judge prompt; no truncation"},
        "internal_validation": {
            "enabled": True, "items": 400, "scenarios": 400, "allow_shared_scenarios": True,
            "partition_unit": "complete_input", "uses_gold_labels_for_selection": False,
            "selection": "One item per scenario; one item per source model in each domain/trigger stratum; exact-input groups intact",
            "item_ids_sha256": _hash(sorted(selected)),
            "checkpoint_selection": "Minimum validation loss; validation excluded from optimizer updates and class counts",
        },
        "distributions": {
            "all": _distribution(prepared),
            **{split: _distribution([row for row in prepared if row["training_split"] == split])
               for split in partitions},
        },
        "source_distributions": {split: source_distribution(members) for split, members in partitions.items()},
        "overlap": overlap,
        "human_label_audit": {"relabelled": 0, "dropped": 0,
                              "six_label_structure_valid": len(prepared),
                              "six_label_consistent_with_rubric": len(prepared)},
        "supervision_contract": {
            "label_fields": list(LABEL_FIELDS), "completion_order": [*LABEL_FIELDS, "rationale"],
            "generation_prefix": "Official enable_thinking=false template",
            "prompt_loss": "All system/user/generation-prefix tokens masked",
            "human_rationale": "Original nonblank comment within schema limit; budget 0.1",
            "absent_rationale": "Only placeholder contents masked; JSON suffix and assistant EOS supervised",
            "truncate": False,
        },
        "interpretation": "Validation of held-out judgments within 400 known scenarios; not unseen-scenario generalization or an independent test score",
        "source_policy": "Repartition all original 8000 annotations, including the former 800-item test partition, as explicitly requested; historical splits/results remain immutable",
    }
    return partitions, prepared, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/judge_bench_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/judge_bench_v4"))
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new empty output directory; existing artifacts are immutable")
    paths = [args.source_dir / (split + ".jsonl") for split in ("train", "test")]
    old_manifest = json.loads((args.source_dir / "split_manifest.json").read_text())
    for split, path in zip(("train", "test"), paths):
        if _file_hash(path) != old_manifest["outputs"][split]["sha256"]:
            raise ValueError("An original frozen source changed")
    rows = [json.loads(line) for path in paths for line in path.open(encoding="utf-8") if line.strip()]
    partitions, prepared, manifest = prepare(rows, args.seed)
    manifest["sources"] = [{"path": str(path.resolve()), "sha256": _file_hash(path)} for path in paths]
    manifest["preparation_source_sha256"] = _file_hash(Path(__file__))
    manifest["selection_source_sha256"] = _file_hash(Path(__file__).with_name("resplit.py"))
    manifest["supervision_source_sha256"] = _file_hash(Path(__file__).with_name("supervision_v3.py"))
    training = args.output_dir / "training"
    training.mkdir(parents=True)
    outputs = {}
    for split, members in partitions.items():
        path = args.output_dir / (split + ".jsonl")
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in members:
                handle.write(_json(row) + "\n")
        outputs[split] = {"file": path.name, "items": len(members), "sha256": _file_hash(path)}
    chat = training / "chat.jsonl"
    with chat.open("w", encoding="utf-8", newline="\n") as handle:
        for row in prepared:
            handle.write(_json(row) + "\n")
    manifest["output"] = {"file": chat.name, "items": len(prepared), "sha256": _file_hash(chat)}
    manifest["outputs"] = outputs
    for path in (args.output_dir / "split_manifest.json", training / "training_manifest.json"):
        path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset_id": DATASET_ID, "fit": 7600, "validation": 400,
                      "scenarios_in_each_split": 400, "full_input_overlap": 0,
                      "class_weights_from": manifest["class_balance"]["computed_from"]}))


if __name__ == "__main__":
    main()
