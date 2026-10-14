"""Build the reproducible, checkpoint-covering 4,000-item human Eval cohort."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

from benchmark_catalog import ROOT_DIR, load_benchmark_items


MODELS = (
    "deepseek-v4-pro", "gemini-3.8-flash", "glm-5.3-flash",
    "gpt-5.6-luna", "grok-4.7",
)
SUBSET_ID = "human-eval-4000-2026-10-02"
SEED = 20261002
RESULT_PATH = ROOT_DIR / "eval_results/human_eval_4000.json.gz"
BATCH2_MODELS = (
    "claude-opus-5-5", "gemini-3.1-pro-preview", "gpt-oss-20b",
    "qwen3.5-9b", "qwen3.8-27b",
)
BATCH2_SUBSET_ID = "human-eval-4000-batch2-2026-10-07"
BATCH2_FILENAME = "human_eval_4000_batch2.json.gz"


def _select(available: dict[str, dict], events: list[str], rng: random.Random,
            models: tuple[str, ...] = MODELS):
    """Match two distinct successful events per model to balanced event quotas."""
    base, extra = divmod(2 * len(models), len(events))
    if base < 1:
        raise ValueError("Ten selections cannot cover this scenario's events")
    quotas = list(itertools.combinations(events, extra))
    rng.shuffle(quotas)
    for added in quotas:
        # source -> model (2) -> event (1) -> sink (event quota)
        sink = 1 + len(models) + len(events)
        graph = [[] for _ in range(sink + 1)]

        def edge(a, b, capacity):
            forward = [b, capacity, len(graph[b])]
            graph[a].append(forward)
            graph[b].append([a, 0, len(graph[a]) - 1])
            return forward

        choices = []
        model_order = list(enumerate(models))
        rng.shuffle(model_order)
        for mi, model in model_order:
            edge(0, 1 + mi, 2)
            event_order = list(enumerate(events))
            rng.shuffle(event_order)
            for ei, event_id in event_order:
                if event_id in available[model]:
                    link = edge(1 + mi, 1 + len(models) + ei, 1)
                    choices.append((model, event_id, link))
        for ei, event_id in enumerate(events):
            edge(1 + len(models) + ei, sink, base + (event_id in added))
        flow = 0
        while flow < 10:
            previous = {0: None}
            queue = deque([0])
            while queue and sink not in previous:
                node = queue.popleft()
                for index, link in enumerate(graph[node]):
                    if link[1] and link[0] not in previous:
                        previous[link[0]] = (node, index)
                        queue.append(link[0])
            if sink not in previous:
                break
            node = sink
            while node:
                parent, index = previous[node]
                link = graph[parent][index]
                link[1] -= 1
                graph[node][link[2]][1] += 1
                node = parent
            flow += 1
        if flow == 10:
            return [(model, event) for model, event, link in choices if not link[1]]
    raise ValueError("Successful responses cannot satisfy the per-scenario quotas")


def build_subset(models: tuple[str, ...] = MODELS,
                 subset_id: str = SUBSET_ID, seed: int = SEED) -> dict:
    if len(models) != 5 or len(set(models)) != 5:
        raise ValueError("A human Eval batch requires five distinct models")
    benchmark = {item["path"]: item for item in load_benchmark_items()}
    runs = defaultdict(dict)
    sources = {}
    for model in models:
        for path in sorted((ROOT_DIR / "eval_results" / model).glob("*.json")):
            raw = path.read_bytes()
            run = json.loads(raw)
            source = run["source_path"]
            if run["model"] != model or source not in benchmark:
                raise ValueError(f"Unexpected result: {path}")
            if model in runs[source]:
                raise ValueError(f"Multiple runs for {model}/{source}; select one explicitly")
            item = benchmark[source]
            if run["scenario"] != item["scenario"] or run["instance"] != item["instance"]:
                raise ValueError(f"Result snapshots differ from the benchmark: {path}")
            relative = path.relative_to(ROOT_DIR).as_posix()
            sources[relative] = hashlib.sha256(raw).hexdigest()
            runs[source][model] = (run, relative)
    if set(runs) != set(benchmark):
        raise ValueError("Results do not cover the entire benchmark")

    rng = random.Random(seed)
    selected = []
    for source in sorted(benchmark):
        if set(runs[source]) != set(models):
            raise ValueError(f"Missing a model's scenario: {source}")
        scenario = benchmark[source]["scenario"]
        events = list(scenario["events"])
        available = {}
        for model, (run, _) in runs[source].items():
            successful = [p for p in run["checkpoints"] if p["agent"]["decision"] == "final"]
            available[model] = {p["event_id"]: p for p in successful}
            if len(available[model]) != len(successful) or set(available[model]) - set(events):
                raise ValueError(f"Duplicate or unknown event in {model}/{source}")
        pairs = _select(available, events, rng, models)
        rng.shuffle(pairs)
        pairs.sort(key=lambda pair: events.index(pair[1]))
        for model, event_id in pairs:
            run, result_path = runs[source][model]
            checkpoint = available[model][event_id]
            selected.append({
                "item_id": f"{run['run_id']}/{event_id}",
                "source_path": source,
                "scenario_key": source,
                "source_result": result_path,
                "event_id": event_id,
                "model": model,
                "domain_id": run["domain_id"],
                "trigger_type": run["trigger_type"],
                "instance": run["instance"],
                "scenario_description": scenario["scenario_description"],
                "event_prefix": [
                    {"event_id": key, **scenario["events"][key]}
                    for key in events[:events.index(event_id) + 1]
                ],
                "agent": checkpoint["agent"],
            })
    expected = {(source, event) for source, item in benchmark.items()
                for event in item["scenario"]["events"]}
    actual = {(item["source_path"], item["event_id"]) for item in selected}
    counts = Counter(item["model"] for item in selected)
    if (len(selected) != 4000 or len({x["item_id"] for x in selected}) != 4000
            or counts != Counter({model: 800 for model in models})
            or actual != expected or len(expected) != 2160):
        raise ValueError("Subset failed the size, uniqueness, balance or coverage checks")
    return {
        "schema_version": 1, "subset_id": subset_id, "seed": seed,
        "models": list(models), "items_per_model": dict(counts),
        "scenario_count": len(benchmark), "checkpoint_count": len(expected),
        "items_per_scenario": 10, "items_per_model_per_scenario": 2,
        "source_sha256": sources, "items": selected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch2", action="store_true",
                        help="Build the additional October 7 cohort; preserve batch one")
    args = parser.parse_args()
    subset = (build_subset(BATCH2_MODELS, BATCH2_SUBSET_ID) if args.batch2
              else build_subset())
    path = (ROOT_DIR / "eval_results" / BATCH2_FILENAME
            if args.batch2 else RESULT_PATH)
    raw = json.dumps(subset, ensure_ascii=False, separators=(",", ":")).encode()
    compressed = gzip.compress(raw, mtime=0)
    if path.exists() and path.read_bytes() != compressed:
        raise SystemExit(f"Refusing to replace a different published subset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    print(json.dumps({"items": len(subset["items"]), "per_model": subset["items_per_model"],
                      "checkpoints": subset["checkpoint_count"], "bytes": len(compressed),
                      "sha256": hashlib.sha256(compressed).hexdigest()}))


if __name__ == "__main__":
    main()
