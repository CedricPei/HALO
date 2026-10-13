"""Validate the published shared-state dataset without model calls or output files.

Run ``python3 -B -m evaluation.validate`` to replay every independent event prefix.
"""

from collections import Counter
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from benchmark_catalog import ROOT_DIR, load_benchmark_items
from command_parser import CommandParser
from creators.prompt import TRIGGER_DEFINITIONS
from creators.validator import _copy_state, _execute_checked, find_structure_conflicts
from env.apps.Wallet import Wallet


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hashes():
    return {
        str(path.relative_to(ROOT_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT_DIR / "domains").rglob("*")
        if path.is_file() and path.name != ".DS_Store"
        and not {"outputs", "logs", "__pycache__"}.intersection(path.parts)
    }


def _initial_conflicts(instance, earliest):
    errors = []
    for filename, collection in (("email", "emails"), ("messaging", "messages"),
                                 ("wallet", "transactions")):
        for row in _read(instance / "app" / f"{filename}.json")[collection]:
            if datetime.fromisoformat(row["timestamp"]) > earliest:
                errors.append(f"Future initial {filename} record: {row['timestamp']}")
    for row in _read(instance / "app/calendar.json")["reminders"]:
        if row.get("completed") and datetime.fromisoformat(
                row.get("due_time") or row["remind_time"]) > earliest:
            errors.append(f"Future completed initial reminder: {row['title']}")
    wallet = _read(instance / "app/wallet.json")
    transactions = sorted(wallet["transactions"],
                          key=lambda row: datetime.fromisoformat(row["timestamp"]))
    effects = [Decimal(str(Wallet._balance_effect(row))) for row in transactions]
    balance = Decimal(str(wallet["wallet"]["balance"])) - sum(effects)
    if abs(balance) > Decimal("0.005") and not (
            instance / "data/wallet_balance_reconciliation.txt").is_file():
        errors.append(f"Undocumented opening wallet balance: {balance}")
    for effect in effects:
        balance += effect
        if balance < Decimal("-0.005"):
            errors.append("Chronological wallet history becomes negative")
            break
    return errors


def _check_effect(command, result):
    parameters = command["parameter"]
    if not isinstance(parameters, dict) or "changes" in parameters:
        raise ValueError("App mutations require a flat parameter object")
    if not isinstance(result, dict):
        raise ValueError("Expected a concrete created/updated record")
    for key, value in parameters.items():
        if value is not None and result.get(key) != value:
            raise ValueError(f"Action did not preserve requested {key}")


def verify():
    before = _source_hashes()
    errors = []
    catalog = load_benchmark_items()
    groups = Counter(item["instance_id"] for item in catalog)
    counts = Counter((item["instance_id"], item["trigger_type"]) for item in catalog)
    if len(catalog) != 400 or len(groups) != 50 or set(groups.values()) != {8}:
        errors.append("Expected 400 scenarios in 50 instances, eight per instance")
    expected_groups = {(instance, trigger) for instance in groups
                       for trigger in TRIGGER_DEFINITIONS}
    if set(counts) != expected_groups or set(counts.values()) != {2}:
        errors.append("Expected two scenarios for each of four triggers per instance")
    if list((ROOT_DIR / "domains").glob("*/instances/*/scenario_state")):
        errors.append("Scenario-specific initial states must not exist")
    earliest, valid_items = {}, []
    for item in catalog:
        try:
            public = item["scenario"]
            if set(public) != {"scenario_description", "events"}:
                raise ValueError("Public scenario must contain only description and events")
            first = datetime.fromisoformat(next(iter(public["events"].values()))["time"])
            envelope = dict(public, scenario_id=item["scenario_id"],
                            trigger_type=item["trigger_type"],
                            reference_time=first.date().isoformat(), ini_data_commands=[])
            conflicts = find_structure_conflicts(envelope, item["trigger_type"])
            if conflicts:
                raise ValueError("; ".join(conflicts))
            instance = item["instance_id"]
            earliest[instance] = min(earliest.get(instance, first), first)
            valid_items.append(item)
        except Exception as error:
            errors.append(f"{item['item_id']}: {error}")

    databases, prefixes = set(), 0
    expected_prefixes = sum(len(item["scenario"]["events"]) for item in valid_items)
    for item in valid_items:
        instance = (ROOT_DIR / item["path"]).parents[2]
        if instance not in databases:
            try:
                database = instance / "data" / f"{instance.name}.sqlite"
                with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
                    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                        errors.append(f"SQLite integrity failed: {instance.name}")
                    if connection.execute("PRAGMA foreign_key_check").fetchall():
                        errors.append(f"SQLite foreign-key failure: {instance.name}")
                errors.extend(f"{instance.name}: {error}" for error in
                              _initial_conflicts(instance, earliest[instance.name]))
            except Exception as error:
                errors.append(f"{instance.name}: {error}")
            databases.add(instance)

        events = list(item["scenario"]["events"].items())
        for length in range(1, len(events) + 1):
            with tempfile.TemporaryDirectory(prefix="HALO-validate-") as temporary:
                parser = None
                try:
                    branch = Path(temporary) / instance.name
                    _copy_state(instance, branch)
                    parser = CommandParser.for_instance(branch, instance.name)
                    for _, event in events[:length]:
                        command = {key: event[key] for key in ("app", "function", "parameter")}
                        result = _execute_checked(parser, command)
                        if event["function"].startswith(("create_", "update_")):
                            _check_effect(command, result)
                    wallet = _read(branch / "app/wallet.json")["wallet"]
                    if wallet is not None and wallet["balance"] < 0:
                        raise ValueError("Negative wallet balance")
                    prefixes += 1
                except Exception as error:
                    errors.append(f"{item['item_id']} prefix {length}: {error}")
                finally:
                    if parser is not None:
                        parser.apps["database"].connection.close()

    after = _source_hashes()
    changed = [path for path in before.keys() | after.keys() if before.get(path) != after.get(path)]
    errors.extend(f"Validation modified a source: {path}" for path in sorted(changed))
    if prefixes != expected_prefixes:
        errors.append(f"Only {prefixes}/{expected_prefixes} independent prefixes executed")
    return {
        "valid": not errors,
        "scenarios": len(catalog), "instances": len(groups),
        "sqlite_databases_checked": len(databases),
        "independent_prefixes_replayed": prefixes,
        "source_files_unchanged": len(before) if not changed else 0,
        "external_model_calls": 0,
        "scope": "Local structure and execution; does not replace semantic review or real model evaluation.",
        "errors": errors,
    }


if __name__ == "__main__":
    try:
        report = verify()
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {"valid": False, "errors": [f"{type(error).__name__}: {error}"]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)
