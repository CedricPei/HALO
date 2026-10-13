"""Create an item holdout with all 400 benchmark scenarios in both splits.

Selection uses scenario/model metadata and exact-input hashes, never gold labels.
Each scenario contributes 18 training and 2 test rows. Identical complete inputs
stay together, including conflicting annotations. Original labels are retained.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import itertools
import json
from pathlib import Path
import random

from .dataset import _distribution, _duplicate_annotation_audit, _hash_overlap
from .prepare_training import _file_hash, _hash, _json


BENCHMARK_ID = "judge-bench-v3-shared-scenarios-20261011"


def select_test_ids(rows: list[dict], seed: int, *, holdout_per_scenario: int = 2) -> set[str]:
    if holdout_per_scenario not in (1, 2):
        raise ValueError("Holdout must contain one or two items per scenario")
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


def prepare(rows: list[dict], seed: int) -> tuple[list[dict], list[dict], dict]:
    selected = select_test_ids(rows, seed)
    train, test = [], []
    for source in sorted(rows, key=lambda r: r["item_id"]):
        row = deepcopy(source)
        split = "test" if row["item_id"] in selected else "train"
        row["source_partition"] = {"benchmark_id": source["benchmark_id"], "split": source["split"]}
        row.update(benchmark_id=BENCHMARK_ID, split=split)
        if _hash(row["input_context"]) != row["hashes"]["input_sha256"]:
            raise ValueError("Original complete-input hash changed")
        (test if split == "test" else train).append(row)
    if len(train) != 7200 or len(test) != 800:
        raise ValueError("Must retain 7200 training and 800 test annotations")
    overlap = {field: _hash_overlap(train, test, field)
               for field in ("input_sha256", "context_sha256", "candidate_action_sha256")}
    if overlap["input_sha256"]["shared_unique_hashes"]:
        raise ValueError("An identical full judge input crosses the split")
    for split, expected in ((train, 18), (test, 2)):
        if len({r["provenance"]["scenario_key"] for r in split}) != 400 or any(
            n != expected for n in Counter(r["provenance"]["scenario_key"] for r in split).values()):
            raise ValueError("Scenario coverage differs")
    overlap.update(scenario_key={"train_unique": 400, "test_unique": 400, "shared_count": 400},
                   interpretation="Within-benchmark item holdout: all scenarios intentionally shared; complete identical inputs remain together. Prefixes/background may be shared across different candidate actions.")
    manifest = {
        "schema_version": 3, "benchmark_id": BENCHMARK_ID, "seed": seed,
        "selection": {"method": "Two rows per scenario; exact-input groups kept intact; two test rows per source model in each domain/trigger stratum",
                      "uses_gold_labels": False, "train_rows_per_scenario": 18, "test_rows_per_scenario": 2,
                      "test_item_ids_sha256": _hash(sorted(selected)), "scenarios": 400},
        "distributions": {"train": _distribution(train), "test": _distribution(test)},
        "overlap": overlap,
        "duplicate_annotation_audit": {"train": _duplicate_annotation_audit(train), "test": _duplicate_annotation_audit(test)},
        "label_policy": "All original human labels and comments preserved; no relabelling or filtering",
        "interpretation": "Specialized judge for these 400 known benchmark scenarios; scores assess held-out items within these scenarios, not unseen-scenario generalization.",
    }
    return train, test, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/judge_bench_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/judge_bench_v3"))
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new empty output directory; previous splits are immutable")
    paths = [args.source_dir / name for name in ("train.jsonl", "test.jsonl")]
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    old_manifest = json.loads((args.source_dir / "split_manifest.json").read_text())
    for name, path in zip(("train", "test"), paths):
        if _file_hash(path) != old_manifest["outputs"][name]["sha256"]:
            raise ValueError("Original source split changed")
    train, test, manifest = prepare(rows, args.seed)
    manifest["sources"] = [{"path": str(p.resolve()), "sha256": _file_hash(p)} for p in paths]
    manifest["context_policy"] = old_manifest["context_policy"]
    manifest["resplit_source_sha256"] = _file_hash(Path(__file__))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, members in (("train", train), ("test", test)):
        path = args.output_dir / (name + ".jsonl")
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in members:
                handle.write(_json(row) + "\n")
        outputs[name] = {"file": path.name, "items": len(members), "sha256": _file_hash(path)}
    manifest["outputs"] = outputs
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train": 7200, "test": 800, "scenarios_in_each_split": 400,
                      "full_input_overlap": 0, "test_labels": manifest["distributions"]["test"]["gold"]}))


if __name__ == "__main__":
    main()
