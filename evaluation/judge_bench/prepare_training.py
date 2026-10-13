"""Prepare traceable SFT chats using only the frozen 7,200-item training split.

The original evaluation prompt and human labels are unchanged. Rationale loss
uses an existing human comment only when it fits the frozen response schema.
Other rows retain a schema-compatible placeholder whose entire suffix is masked.
No test-set file is opened or used to choose the internal validation split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from . import prompt


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT_DIR / "outputs/judge_bench_v1/train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs/judge_bench_v1/training"
DATASET_ID = "judge-bench-v1-sft-20261011"
SEED = 20261011
VALIDATION_SCENARIOS = 18
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


def _select_validation(rows: list[dict], seed: int) -> list[str]:
    """Sample 18 distinct domain/trigger strata with balanced marginal quotas."""
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    scenarios: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        meta = row["provenance"]
        groups[meta["domain_id"], meta["trigger_type"]].add(meta["scenario_key"])
        scenarios[meta["scenario_key"]].append(row)
    domains = sorted({domain for domain, _ in groups})
    triggers = sorted({trigger for _, trigger in groups})
    models = {row["provenance"]["source_model"] for row in rows}
    if (
        len(rows) != 7200 or len({row["item_id"] for row in rows}) != 7200
        or any(row.get("split") != "train" for row in rows)
        or len(scenarios) != 360 or len(groups) != 40
        or len(domains) != 10 or len(triggers) != 4 or len(models) != 10
        or any(len(keys) != 9 for keys in groups.values())
    ):
        raise ValueError("Input must be the frozen 7,200-row training split")
    for key, members in scenarios.items():
        counts = Counter(row["provenance"]["source_model"] for row in members)
        if len(members) != 20 or counts != Counter({model: 2 for model in models}):
            raise ValueError(f"Unexpected scenario model balance: {key}")
    rng = random.Random(seed)
    domains_with_two = set(rng.sample(domains, 8))
    triggers_with_five = set(rng.sample(triggers, 2))
    domain_quotas = {domain: 1 + (domain in domains_with_two) for domain in domains}
    trigger_quotas = {trigger: 4 + (trigger in triggers_with_five) for trigger in triggers}

    # Integral max flow chooses a domain/trigger cell at most once. Randomized
    # edge order is deterministic for this seed; all decisions use metadata only.
    sink = 1 + len(domains) + len(triggers)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def edge(source: int, target: int, capacity: int) -> list[int]:
        link = [target, capacity, len(graph[target])]
        graph[source].append(link)
        graph[target].append([source, 0, len(graph[source]) - 1])
        return link

    choices = []
    domain_order = list(enumerate(domains))
    rng.shuffle(domain_order)
    for index, domain in domain_order:
        edge(0, 1 + index, domain_quotas[domain])
        trigger_order = list(enumerate(triggers))
        rng.shuffle(trigger_order)
        for ti, trigger in trigger_order:
            choices.append((domain, trigger, edge(1 + index, 1 + len(domains) + ti, 1)))
    for ti, trigger in enumerate(triggers):
        edge(1 + len(domains) + ti, sink, trigger_quotas[trigger])
    flow = 0
    while flow < VALIDATION_SCENARIOS:
        previous = {0: None}
        queue = deque([0])
        while queue and sink not in previous:
            source = queue.popleft()
            for index, link in enumerate(graph[source]):
                if link[1] and link[0] not in previous:
                    previous[link[0]] = source, index
                    queue.append(link[0])
        if sink not in previous:
            raise ValueError("Could not satisfy validation scenario quotas")
        target = sink
        while target:
            source, index = previous[target]
            link = graph[source][index]
            link[1] -= 1
            graph[target][link[2]][1] += 1
            target = source
        flow += 1
    strata = sorted((domain, trigger) for domain, trigger, link in choices if link[1] == 0)
    return sorted(rng.choice(sorted(groups[stratum])) for stratum in strata)


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


def _distribution(rows: list[dict]) -> dict:
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


def prepare(rows: list[dict], *, seed: int = SEED) -> tuple[list[dict], dict]:
    """Prepare all 7,200 rows and a label-blind internal scenario holdout."""
    validation_keys = _select_validation(rows, seed)
    validation_set = set(validation_keys)
    prepared = [
        _prepare_row(row, "validation" if row["provenance"]["scenario_key"] in validation_set else "train")
        for row in rows
    ]
    fit = [row for row in prepared if row["training_split"] == "train"]
    validation = [row for row in prepared if row["training_split"] == "validation"]
    if len(fit) != 6840 or len(validation) != 360:
        raise ValueError("Internal split must contain 6,840 fit and 360 validation rows")
    for members, expected in ((fit, 684), (validation, 36)):
        if any(count != expected for count in Counter(row["provenance"]["source_model"] for row in members).values()):
            raise ValueError("Internal split lost source-model balance")
    fit_scenarios = {row["provenance"]["scenario_key"] for row in fit}
    val_scenarios = {row["provenance"]["scenario_key"] for row in validation}
    fit_inputs = {row["hashes"]["source_input_context_sha256"] for row in fit}
    val_inputs = {row["hashes"]["source_input_context_sha256"] for row in validation}
    fit_instances = {row["provenance"]["instance_key"] for row in fit}
    val_instances = {row["provenance"]["instance_key"] for row in validation}
    source_groups: dict[str, list[dict]] = defaultdict(list)
    for row in prepared:
        source_groups[row["hashes"]["source_input_context_sha256"]].append(row)
    conflicting = [
        group for group in source_groups.values()
        if len({_hash({field: row["gold"][field] for field in LABEL_FIELDS}) for row in group}) > 1
    ]
    if fit_scenarios & val_scenarios or fit_inputs & val_inputs:
        raise ValueError("Internal training/validation scenario or exact-input overlap")
    manifest = {
        "schema_version": 1, "dataset_id": DATASET_ID, "seed": seed,
        "rows": 7200, "test_set_used": False,
        "prompt": {
            "version": prompt.PROMPT_VERSION, "sha256": prompt.PROMPT_SHA256,
            "source_sha256": _file_hash(Path(prompt.__file__)),
            "context_policy": "Original frozen build_messages(input_context), unchanged full evidence",
        },
        "internal_validation": {
            "selection": "Metadata-only balanced marginal quotas, randomized integral domain/trigger matching, then one scenario per chosen stratum",
            "uses_gold_labels": False, "scenarios": VALIDATION_SCENARIOS,
            "scenario_keys": validation_keys, "scenario_keys_sha256": _hash(validation_keys),
            "domain_scenario_quotas": "8 domains have 2; 2 domains have 1",
            "trigger_scenario_quotas": "2 triggers have 5; 2 triggers have 4",
            "distinct_domain_trigger_strata": 18,
            "items_per_scenario": 20,
        },
        "training_modes": {
            "default": "Fit 6840 training_split=train; validate on 360 training_split=validation",
            "final_all": "Use all 7200 rows only when selected explicitly; no internal validation then",
        },
        "supervision_contract": {
            "label_fields": list(LABEL_FIELDS),
            "completion_order": [*LABEL_FIELDS, "rationale"],
            "loss_prefix_end_units": "Unicode character offset into completion, end exclusive; never UTF-8 bytes",
            "prompt_loss": "All system/user and pre-assistant tokens are masked with -100",
            "human_rationale": "Existing nonblank human comment of at most 1000 characters, unchanged; supervise full completion and EOS",
            "missing_or_overlong_rationale": "Schema-valid underscore placeholder is not a human rationale. Supervise only through the final rationale opening quote; mask placeholder, remaining JSON suffix, and EOS",
            "boundary_tokens": "Mask any token crossing loss_prefix_end; no token may partially supervise the placeholder",
            "truncate": False,
            "audit_only_fields": ["gold", "provenance", "hashes", "supervision", "item_id"],
        },
        "human_label_audit": {
            "six_label_structure_valid": 7200, "six_label_consistent_with_rubric": 7200,
            "relabelled": 0, "dropped": 0,
            "rationale_sources": _counts(row["supervision"]["rationale_source"] for row in prepared),
            "conflicting_exact_input_groups": len(conflicting),
            "conflicting_group_item_ids": [[row["item_id"] for row in group] for group in conflicting],
            "human_comment_quality": "Original annotation comments are weak rationale supervision; no new reasoning, adjudication, shortening, or factual rewriting",
        },
        "distributions": {
            "all": _distribution(prepared), "train": _distribution(fit),
            "validation": _distribution(validation),
        },
        "overlap": {
            "scenario_count": len(fit_scenarios & val_scenarios),
            "exact_input_count": len(fit_inputs & val_inputs),
            "shared_instance_count": len(fit_instances & val_instances),
            "interpretation": "Whole-scenario internal holdout; user instances and reusable background may overlap",
        },
    }
    return prepared, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    with args.input.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    prepared, manifest = prepare(rows, seed=args.seed)
    manifest["source_train"] = {
        "path": str(args.input.resolve()), "sha256": _file_hash(args.input), "items": len(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chat_path = args.output_dir / "chat.jsonl"
    temporary = chat_path.with_suffix(".jsonl.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in prepared:
                handle.write(_json(row) + "\n")
        temporary.replace(chat_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest["output"] = {"file": chat_path.name, "items": len(prepared), "sha256": _file_hash(chat_path)}
    path = args.output_dir / "training_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({
        "dataset_id": DATASET_ID, "rows": len(prepared), "fit": 6840, "validation": 360,
        "rationale_sources": manifest["human_label_audit"]["rationale_sources"],
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
