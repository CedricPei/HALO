"""Read-only loading of benchmark items and evaluation checkpoints."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DOMAINS_DIR = ROOT_DIR / "domains"
RESULTS_DIR = ROOT_DIR / "eval_results"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def input_fingerprint(item: dict) -> str:
    """Identify the published scenario, one shared state, and execution code."""
    instance_dir = (ROOT_DIR / item["path"]).parents[2]
    digest = hashlib.sha256()
    digest.update(json.dumps(item["scenario"], sort_keys=True).encode())
    paths = [ROOT_DIR / item["path"], instance_dir / "profile.json"]
    for folder in ("data", "app"):
        paths.extend(path for path in (instance_dir / folder).rglob("*") if path.is_file())
    paths.extend(ROOT_DIR / name for name in (
        "benchmark_catalog.py", "command_parser.py", "evaluation/runner.py", "evaluation/prompt.py",
        "evaluation/intensity.py", "llm_client.py", "env/app_surfaces/evaluated_agent.md",
    ))
    paths.extend((ROOT_DIR / "env/apps").glob("*.py"))
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT_DIR)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _state_summary(instance_dir: Path, instance_id: str) -> dict:
    data_dir = instance_dir / "data"
    database_path = data_dir / f"{instance_id}.sqlite"
    database = {}
    if database_path.exists():
        connection = sqlite3.connect(database_path)
        try:
            tables = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            for table in tables:
                columns = [
                    row[1]
                    for row in connection.execute(f'PRAGMA table_info("{table}")')
                ]
                rows = connection.execute(
                    f'SELECT * FROM "{table}" LIMIT 50'
                ).fetchall()
                database[table] = {"columns": columns, "rows": rows}
        finally:
            connection.close()

    files = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(data_dir.glob("*.txt"))
    }
    apps = {
        path.stem: _json(path)
        for path in sorted((instance_dir / "app").glob("*.json"))
    }
    return {"database": database, "files": files, "apps": apps}


def load_benchmark_items() -> list[dict]:
    """Return the current published Scenarios and their shared Instance state."""
    items = []
    profile_cache: dict[Path, dict] = {}
    state_cache: dict[Path, dict] = {}
    for public_path in sorted(DOMAINS_DIR.glob("*/instances/*/scenarios/*/*.json")):
        instance_dir = public_path.parents[2]
        instance_id = instance_dir.name
        domain_dir = instance_dir.parents[1]
        trigger = public_path.parent.name
        scenario_id = public_path.stem
        scenario = _json(public_path)
        if set(scenario) != {"scenario_description", "events"}:
            raise ValueError(f"Scenario must use its shared Instance state: {public_path}")
        if instance_dir not in state_cache:
            state_cache[instance_dir] = _state_summary(instance_dir, instance_id)
            profile_cache[instance_dir] = _json(instance_dir / "profile.json")
        items.append(
            {
                "item_id": f"{domain_dir.name}/{instance_id}/{scenario_id}",
                "domain_id": domain_dir.name,
                "instance_id": instance_id,
                "scenario_id": scenario_id,
                "trigger_type": trigger,
                "instance": profile_cache[instance_dir],
                "scenario": scenario,
                "state": state_cache[instance_dir],
                "path": str(public_path.relative_to(ROOT_DIR)),
            }
        )
    return items


def load_evaluation_items() -> list[dict]:
    """Flatten saved model runs into checkpoint-level review items."""
    items = []
    for path in sorted(RESULTS_DIR.glob("*/*.json")):
        run = _json(path)
        source_path = run["source_path"]
        if "scenario" not in run or "instance" not in run:
            raise ValueError(f"Evaluation result lacks scenario/profile snapshots; rerun: {path}")
        scenario = run["scenario"]
        instance = run["instance"]
        events = list(scenario["events"].items())
        event_positions = {
            event_id: index for index, (event_id, _) in enumerate(events)
        }
        for checkpoint in run["checkpoints"]:
            if checkpoint["agent"]["decision"] == "error":
                continue
            event_id = checkpoint["event_id"]
            items.append(
                {
                    "item_id": f"{run['run_id']}/{event_id}",
                    "source_path": source_path,
                    "scenario_key": source_path,
                    "model": run["model"],
                    "domain_id": run["domain_id"],
                    "trigger_type": run["trigger_type"],
                    "instance": instance,
                    "scenario_description": scenario["scenario_description"],
                    "event_prefix": [
                        {"event_id": prefix_id, **event}
                        for prefix_id, event in events[
                            : event_positions[event_id] + 1
                        ]
                    ],
                    "agent": checkpoint["agent"],
                }
            )
    return items


def evaluation_groups(items: list[dict]) -> list[list[str]]:
    """Keep all checkpoints from one Scenario assigned to the same person."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in items:
        grouped[item["scenario_key"]].append(item["item_id"])
    return [grouped[key] for key in sorted(grouped)]
