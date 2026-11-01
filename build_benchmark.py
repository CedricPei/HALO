"""Internal benchmark construction driver (not an evaluated-agent CLI)."""

import argparse
import hashlib
import json
import logging
import random
import sqlite3
import time
from collections import Counter
from concurrent.futures import CancelledError
from itertools import combinations
from pathlib import Path
from threading import Event

from creators import InstanceCreator, NoiseCreator, build_scenario
from creators.prompt import TRIGGER_DEFINITIONS
from creators.scenario_creator import load_published_scenario
from creators.validator import (
    description_similarity,
    validate_published_scenario,
)
from llm_client import LLMTransportError


ROOT_DIR = Path(__file__).resolve().parent
PLAN_PATH = ROOT_DIR / "benchmark_plan.json"
DOMAINS_DIR = ROOT_DIR / "domains"
BENCHMARK_DIR = ROOT_DIR / "benchmark"
LOG_PATH = BENCHMARK_DIR / "build.log"
LOGGER = logging.getLogger(__name__)
SCENARIOS_PER_TRIGGER = 2


def load_plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def instance_schedule(plan, domain_spec):
    return [
        {"instance_id": f"{domain_spec['domain_id']}_{index:02d}"}
        for index in range(1, plan["instances_per_domain"] + 1)
    ]


def scenario_schedule():
    order = list(TRIGGER_DEFINITIONS) * SCENARIOS_PER_TRIGGER
    random.shuffle(order)
    return order


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _configure_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")],
        force=True,
    )


def _prepare_domains(domain_specs):
    """Materialize the selected human-defined Domain folders without an LLM call."""
    DOMAINS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in domain_specs:
        domain_dir = DOMAINS_DIR / spec["domain_id"]
        (domain_dir / "instances").mkdir(parents=True, exist_ok=True)
        _write_json(domain_dir / "domain.json", spec)


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shared_state_errors(instance_dir, instance_id):
    """Check the materialized evaluation state, independently of build logs."""
    errors = []
    try:
        profile = _load_json(instance_dir / "profile.json")
        for field in ("instance_description", "personal_description"):
            if not isinstance(profile.get(field), str) or not profile[field].strip():
                errors.append(f"profile.json has no {field}")
    except Exception as error:
        errors.append(f"unreadable profile.json: {error}")
    for app in ("calendar", "contacts", "email", "messaging", "wallet"):
        try:
            if not isinstance(_load_json(instance_dir / "app" / f"{app}.json"), dict):
                errors.append(f"app/{app}.json must be an object")
        except Exception as error:
            errors.append(f"unreadable app/{app}.json: {error}")
    database_path = instance_dir / "data" / f"{instance_id}.sqlite"
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                errors.append("shared database failed integrity_check")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                errors.append("shared database failed foreign_key_check")
        finally:
            connection.close()
        for path in (instance_dir / "data").glob("*.txt"):
            path.read_text(encoding="utf-8")
    except Exception as error:
        errors.append(f"unreadable shared data: {error}")
    return errors


def _published_instance(instance_dir, instance_id):
    """Validate every published branch against the same final Instance state."""
    errors = _shared_state_errors(instance_dir, instance_id)
    scenarios, trigger_counts = [], Counter()
    expected_paths = set()
    for trigger in TRIGGER_DEFINITIONS:
        trigger_dir = instance_dir / "scenarios" / trigger
        public_paths = sorted(trigger_dir.glob("*.json"))
        expected_paths.update(
            trigger_dir / f"{trigger}_{slot}.json"
            for slot in range(1, SCENARIOS_PER_TRIGGER + 1)
        )
        trigger_counts[trigger] = len(public_paths)
        if len(public_paths) != SCENARIOS_PER_TRIGGER:
            errors.append(
                f"{trigger} has {len(public_paths)} Scenarios; expected {SCENARIOS_PER_TRIGGER}"
            )
        for public_path in public_paths:
            try:
                scenario = load_published_scenario(public_path)
                conflicts = validate_published_scenario(
                    instance_dir, instance_id, scenario, trigger
                )
                errors.extend(f"{public_path.stem}: {item}" for item in conflicts)
                if not conflicts:
                    scenarios.append((public_path, scenario))
            except Exception as error:
                errors.append(f"{public_path.stem}: {type(error).__name__}: {error}")
    actual_paths = {path for path, _ in scenarios}
    if actual_paths != expected_paths:
        errors.append("Published Scenario paths do not match the eight scheduled slots")
    return errors, scenarios, trigger_counts


def _uses_published_state(instance_dir):
    """An archived or fully completed Instance must not be constructed again."""
    if not instance_dir.exists():
        return False
    if not (instance_dir / "outputs" / "instance.json").exists():
        return True
    return (instance_dir / "outputs" / "noise.json").exists() and all(
        (instance_dir / "scenarios" / trigger / f"{trigger}_{slot}.json").exists()
        for trigger in TRIGGER_DEFINITIONS
        for slot in range(1, SCENARIOS_PER_TRIGGER + 1)
    )


def audit_benchmark(domain_ids=None, *, write_manifest=True):
    """Validate the current dataset; archived build outcomes are provenance only."""
    plan = load_plan()
    domains_dir, output_dir = DOMAINS_DIR, BENCHMARK_DIR
    errors, warnings, domain_entries, all_records = [], [], [], []
    event_counts = Counter()

    domain_specs = [
        item
        for item in plan["domains"]
        if domain_ids is None or item["domain_id"] in domain_ids
    ]
    expected_domains = {item["domain_id"] for item in domain_specs}
    unknown = set(domain_ids or []) - {
        item["domain_id"] for item in plan["domains"]
    }
    if unknown:
        raise ValueError(f"unknown Domain ids: {sorted(unknown)}")
    actual_domains = (
        {path.name for path in domains_dir.iterdir() if path.is_dir()}
        if domains_dir.exists()
        else set()
    )
    checked_domains = actual_domains if domain_ids is None else actual_domains & expected_domains
    if checked_domains != expected_domains:
        errors.append(
            "Domain directories differ: "
            f"missing={sorted(expected_domains - actual_domains)}, "
            f"unexpected={sorted(checked_domains - expected_domains)}"
        )

    for domain_spec in domain_specs:
        domain_id = domain_spec["domain_id"]
        domain_dir = domains_dir / domain_id
        domain_errors, instance_entries = [], []
        if not (domain_dir / "domain.json").exists():
            domain_errors.append("domain.json is missing")

        for instance_spec in instance_schedule(plan, domain_spec):
            instance_id = instance_spec["instance_id"]
            instance_dir = domain_dir / "instances" / instance_id
            instance_errors, scenario_entries, records = [], [], []
            trigger_counts = Counter()
            if not instance_dir.exists():
                instance_errors.append("Instance directory is missing")
            else:
                instance_errors, scenarios, trigger_counts = _published_instance(
                    instance_dir, instance_id
                )
                for public_path, scenario in scenarios:
                    scenario_id, trigger = public_path.stem, public_path.parent.name
                    event_count = len(scenario["events"])
                    event_counts[event_count] += 1
                    scenario_entries.append(
                        {
                            "scenario_id": scenario_id,
                            "trigger_type": trigger,
                            "event_count": event_count,
                            "path": str(public_path.relative_to(ROOT_DIR)),
                            "sha256": _hash_file(public_path),
                        }
                    )
                    record = {
                        "domain_id": domain_id,
                        "instance_id": instance_id,
                        "scenario_id": scenario_id,
                        "description": scenario["scenario_description"],
                    }
                    records.append(record)
                    all_records.append(record)

            for left, right in combinations(records, 2):
                score = description_similarity(
                    left["description"], right["description"]
                )
                if score >= 0.55:
                    warnings.append(
                        f"possible duplicate {instance_id}/{left['scenario_id']} "
                        f"vs {right['scenario_id']} ({score:.2f})"
                    )
            instance_entries.append(
                {
                    "instance_id": instance_id,
                    "valid": not instance_errors,
                    "errors": instance_errors,
                    "trigger_counts": {
                        key: trigger_counts[key] for key in TRIGGER_DEFINITIONS
                    },
                    "scenarios": scenario_entries,
                }
            )
            errors.extend(
                f"{domain_id}/{instance_id}: {item}" for item in instance_errors
            )

        domain_entries.append(
            {
                "domain_id": domain_id,
                "title": domain_spec["title"],
                "split": domain_spec["split"],
                "valid": not domain_errors
                and all(item["valid"] for item in instance_entries),
                "errors": domain_errors,
                "instances": instance_entries,
            }
        )
        errors.extend(f"{domain_id}: {item}" for item in domain_errors)

    manifest = {
        "benchmark_id": plan["benchmark_id"],
        "valid": not errors,
        "summary": {
            "domain_count": len(checked_domains),
            "instance_count": sum(
                len(item["instances"]) for item in domain_entries
            ),
            "scenario_count": len(all_records),
            "event_count": sum(
                count * frequency for count, frequency in event_counts.items()
            ),
            "event_count_distribution": dict(sorted(event_counts.items())),
        },
        "review_report": {"errors": errors, "warnings": warnings},
        "domains": domain_entries,
    }
    if write_manifest:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _create_instance(domain_spec, instance_spec, benchmark_year, stop_event):
    domain_id = domain_spec["domain_id"]
    instance_id = instance_spec["instance_id"]
    instance_dir = DOMAINS_DIR / domain_id / "instances" / instance_id
    if instance_dir.exists():
        LOGGER.info(f"[{domain_id}/{instance_id}] Instance exists; resuming")
        return
    creator = InstanceCreator()
    retry_feedback = []
    for attempt in range(1, 6):
        if stop_event.is_set():
            raise CancelledError
        try:
            LOGGER.info(
                f"[{domain_id}/{instance_id}] creating Instance "
                f"({attempt}/5)"
            )
            creator.create_instance(
                domain_spec,
                instance_id,
                benchmark_year,
                retry_feedback=retry_feedback,
            )
            return
        except LLMTransportError as error:
            stop_event.set()
            LOGGER.info(
                f"[{domain_id}/{instance_id}] LLM requests exhausted; "
                f"aborting benchmark: {error}"
            )
            raise
        except Exception as error:
            last_error = error
            retry_feedback.append(f"{type(error).__name__}: {error}")
            LOGGER.info(
                f"[{domain_id}/{instance_id}] Instance failed: "
                f"{type(error).__name__}: {error}"
            )
    raise RuntimeError(f"Instance creation failed: {last_error}")


def _build_one_instance(plan, domain_spec, instance_spec, stop_event):
    domain_id, instance_id = domain_spec["domain_id"], instance_spec["instance_id"]
    instance_dir = DOMAINS_DIR / domain_id / "instances" / instance_id
    if stop_event.is_set():
        raise CancelledError
    if _uses_published_state(instance_dir):
        errors, scenarios, _ = _published_instance(instance_dir, instance_id)
        if errors:
            raise ValueError(
                "Existing published Instance failed validation; no state was changed: "
                + "; ".join(errors)
            )
        LOGGER.info(f"[{domain_id}/{instance_id}] Published state validated; skipping construction")
        return {
            "domain_id": domain_id,
            "instance_id": instance_id,
            "status": "complete",
            "elapsed_seconds": 0,
            "scenario_results": [
                {"scenario_id": scenario["scenario_id"], "status": "existing"}
                for _, scenario in scenarios
            ],
            "noise_result": {"status": "materialized"},
            "review": {
                "valid": True,
                "issues": [],
                "source": "local validation of published state",
            },
        }
    _create_instance(
        domain_spec,
        instance_spec,
        plan["benchmark_year"],
        stop_event,
    )
    construction_started = (
        instance_dir / "outputs" / "instance.json"
    ).stat().st_mtime
    scenario_results = []
    order = scenario_schedule()
    trigger_slots = {trigger: 0 for trigger in order}
    for trigger in order:
        if stop_event.is_set():
            raise CancelledError
        trigger_slots[trigger] += 1
        slot = trigger_slots[trigger]
        scenario_id = f"{trigger}_{slot}"
        public_path = instance_dir / "scenarios" / trigger / f"{scenario_id}.json"
        if public_path.exists():
            LOGGER.info(f"[{domain_id}/{instance_id}] {scenario_id} exists; skipping")
            scenario_results.append({"scenario_id": scenario_id, "status": "existing"})
            continue
        LOGGER.info(
            f"[{domain_id}/{instance_id}] generating {trigger} Scenario "
            f"(slot {slot})"
        )
        result = build_scenario(
            domain_id,
            instance_id,
            trigger,
            slot,
            plan["benchmark_year"],
            stop_event=stop_event,
        )
        scenario_id = result["scenario_id"]
        if result["status"] == "accepted":
            LOGGER.info(
                f"[{domain_id}/{instance_id}] accepted {scenario_id} after "
                f"{result['attempts']} attempt(s)"
            )
        else:
            LOGGER.info(
                f"[{domain_id}/{instance_id}] unresolved {scenario_id} after "
                f"{result['attempts']} Creator output attempt(s)"
            )
        scenario_results.append(
            {
                "scenario_id": scenario_id,
                "status": result["status"],
                "attempts": result["attempts"],
                "rejected": result["rejected"],
            }
        )

    skipped = [
        item["scenario_id"]
        for item in scenario_results
        if item["status"] not in {"accepted", "existing"}
    ]
    if skipped:
        review_path = instance_dir / "outputs" / "review.json"
        review = (
            json.loads(review_path.read_text(encoding="utf-8"))
            if review_path.exists()
            else {"valid": False, "issues": ["No Scenario was accepted"]}
        )
        return {
            "domain_id": domain_id,
            "instance_id": instance_id,
            "status": "incomplete",
            "elapsed_seconds": round(time.time() - construction_started, 2),
            "scenario_results": scenario_results,
            "noise_result": {
                "status": "skipped",
                "reason": "Scenario slots were not completed",
            },
            "review": review,
        }

    if stop_event.is_set():
        raise CancelledError
    noise_path = instance_dir / "outputs" / "noise.json"
    if noise_path.exists():
        LOGGER.info(f"[{domain_id}/{instance_id}] Noise exists; skipping")
        existing_noise = json.loads(noise_path.read_text(encoding="utf-8"))
        noise_result = {
            "status": "existing",
            "command_count": len(existing_noise["noise_commands"]),
        }
    else:
        noise_creator = NoiseCreator()
        LOGGER.info(f"[{domain_id}/{instance_id}] generating Noise")
        created_noise = noise_creator.create_noise(
            domain_id, instance_id, stop_event=stop_event
        )
        noise_result = {
            "status": "generated",
            "command_count": len(created_noise["noise_commands"]),
        }

    review_path = instance_dir / "outputs" / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    completion_time = noise_path.stat().st_mtime
    return {
        "domain_id": domain_id,
        "instance_id": instance_id,
        "status": "complete" if review.get("valid") else "review_failed",
        "elapsed_seconds": round(completion_time - construction_started, 2),
        "scenario_results": scenario_results,
        "noise_result": noise_result,
        "review": review,
    }


def build_benchmark(
    domain_ids=None,
    instance_ids=None,
):
    plan = load_plan()
    selected_domains = [
        spec
        for spec in plan["domains"]
        if domain_ids is None or spec["domain_id"] in domain_ids
    ]
    known_domains = {spec["domain_id"] for spec in plan["domains"]}
    unknown_domains = set(domain_ids or []) - known_domains
    if unknown_domains:
        raise ValueError(f"unknown Domain ids: {sorted(unknown_domains)}")
    known_instances = {
        instance_spec["instance_id"]
        for domain_spec in selected_domains
        for instance_spec in instance_schedule(plan, domain_spec)
    }
    unknown_instances = set(instance_ids or []) - known_instances
    if unknown_instances:
        raise ValueError(f"unknown Instance ids: {sorted(unknown_instances)}")

    selected_instances = [
        DOMAINS_DIR / spec["domain_id"] / "instances" / instance["instance_id"]
        for spec in selected_domains
        for instance in instance_schedule(plan, spec)
        if instance_ids is None or instance["instance_id"] in instance_ids
    ]
    published_only = bool(selected_instances) and all(
        _uses_published_state(path) for path in selected_instances
    )
    if published_only:
        logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    else:
        _configure_logging()
        _prepare_domains(selected_domains)
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results, failures = [], []
    stop_event = Event()
    selected_instance_count = 0
    for domain_spec in selected_domains:
        for instance_spec in instance_schedule(plan, domain_spec):
            instance_id = instance_spec["instance_id"]
            if instance_ids is not None and instance_id not in instance_ids:
                continue
            selected_instance_count += 1
            domain_id = domain_spec["domain_id"]
            try:
                results.append(
                    _build_one_instance(
                        plan, domain_spec, instance_spec, stop_event
                    )
                )
            except CancelledError:
                continue
            except LLMTransportError:
                stop_event.set()
                LOGGER.info("LLM requests exhausted; aborting the entire build")
                raise
            except Exception as error:
                failure = {
                    "domain_id": domain_id,
                    "instance_id": instance_id,
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                LOGGER.info(
                    f"[{domain_id}/{instance_id}] BUILD FAILED: "
                    f"{failure['error']}"
                )

    run_elapsed = round(time.monotonic() - started, 2)
    report = {
        "complete": not failures
        and all(item["status"] == "complete" for item in results),
        "run_elapsed_seconds": run_elapsed,
        "construction_elapsed_seconds": sum(
            item["elapsed_seconds"] for item in results
        ),
        "selected_instance_count": selected_instance_count,
        "results": sorted(results, key=lambda item: item["instance_id"]),
        "failures": sorted(failures, key=lambda item: item["instance_id"]),
    }
    LOGGER.info(
        "Build complete | "
        f"status={'complete' if report['complete'] else 'incomplete'} | "
        f"instances={report['selected_instance_count']} | "
        f"elapsed={report['run_elapsed_seconds']}s"
    )

    complete_domain_selection = instance_ids is None
    if complete_domain_selection and report["complete"]:
        audit = audit_benchmark(
            domain_ids=[spec["domain_id"] for spec in selected_domains],
            write_manifest=not published_only,
        )
    else:
        audit = None
    return report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", dest="domain_ids")
    parser.add_argument("--instance", action="append", dest="instance_ids")
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Validate published data and print JSON without writing build artifacts",
    )
    args = parser.parse_args()
    if args.audit_only:
        if args.instance_ids:
            parser.error("--audit-only supports --domain, not --instance")
        audit = audit_benchmark(domain_ids=args.domain_ids, write_manifest=False)
        summary = {
            key: audit[key] for key in ("valid", "summary", "review_report")
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(0 if audit["valid"] else 1)
    report, audit = build_benchmark(
        domain_ids=args.domain_ids,
        instance_ids=args.instance_ids,
    )
    valid = report["complete"] and (audit is None or audit["valid"])
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
