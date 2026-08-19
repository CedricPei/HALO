"""Internal benchmark construction driver (not an evaluated-agent CLI)."""

import argparse
import hashlib
import json
import logging
import random
import time
from collections import Counter
from concurrent.futures import CancelledError
from itertools import combinations
from pathlib import Path
from threading import Event

from creators import InstanceCreator, NoiseCreator, build_scenario
from creators.prompt import TRIGGER_DEFINITIONS
from creators.validator import (
    description_similarity,
    find_noise_structure_conflicts,
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


def audit_benchmark(domain_ids=None):
    """Validate a complete benchmark and write its canonical manifest."""
    plan = load_plan()
    domains_dir, output_dir = DOMAINS_DIR, BENCHMARK_DIR
    errors, warnings, domain_entries, all_records = [], [], [], []
    event_counts, ini_app_counts = Counter(), Counter()
    noise_command_count = 0

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
    if actual_domains != expected_domains:
        errors.append(
            "Domain directories differ: "
            f"missing={sorted(expected_domains - actual_domains)}, "
            f"unexpected={sorted(actual_domains - expected_domains)}"
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
                try:
                    output = _load_json(instance_dir / "outputs" / "instance.json")
                    if output.get("instance_id") != instance_id:
                        instance_errors.append("outputs/instance.json has wrong ID")
                    _load_json(instance_dir / "profile.json")
                except Exception as error:
                    instance_errors.append(f"unreadable Instance metadata: {error}")

                for trigger in TRIGGER_DEFINITIONS:
                    trigger_dir = instance_dir / "scenarios" / trigger
                    public_paths = sorted(trigger_dir.glob(f"{trigger}_*.json"))
                    trigger_counts[trigger] = len(public_paths)
                    if len(public_paths) != SCENARIOS_PER_TRIGGER:
                        instance_errors.append(
                            f"{trigger} has {len(public_paths)} Scenarios; expected "
                            f"{SCENARIOS_PER_TRIGGER}"
                        )
                    for public_path in public_paths:
                        scenario_id = public_path.stem
                        log_dir = instance_dir / "scenarios" / "logs" / scenario_id
                        required = (
                            "scenario_output.json",
                            "validation_output.json",
                            "review_output.json",
                        )
                        missing = [
                            name for name in required if not (log_dir / name).exists()
                        ]
                        if missing:
                            instance_errors.append(f"{scenario_id} missing {missing}")
                            continue
                        try:
                            scenario = _load_json(log_dir / "scenario_output.json")
                            validation = _load_json(
                                log_dir / "validation_output.json"
                            )
                            review = _load_json(log_dir / "review_output.json")
                            public = _load_json(public_path)
                        except Exception as error:
                            instance_errors.append(
                                f"{scenario_id} unreadable: {error}"
                            )
                            continue
                        if validation != {"valid": True, "conflicts": []}:
                            instance_errors.append(f"{scenario_id} failed Validator")
                        if not review.get("valid") or review.get("issues"):
                            instance_errors.append(f"{scenario_id} failed Reviewer")
                        if public != {
                            "scenario_description": scenario.get(
                                "scenario_description"
                            ),
                            "events": scenario.get("events"),
                        }:
                            instance_errors.append(
                                f"{scenario_id} public output differs"
                            )
                        for conflict in validate_published_scenario(
                            instance_dir,
                            instance_id,
                            scenario,
                            trigger,
                        ):
                            instance_errors.append(f"{scenario_id}: {conflict}")

                        ini_apps = sorted(
                            {item["app"] for item in scenario["ini_data_commands"]}
                        )
                        ini_app_counts.update(ini_apps)
                        event_count = len(scenario["events"])
                        event_counts[event_count] += 1
                        scenario_entries.append(
                            {
                                "scenario_id": scenario_id,
                                "trigger_type": trigger,
                                "event_count": event_count,
                                "ini_data_command_count": len(
                                    scenario["ini_data_commands"]
                                ),
                                "ini_data_apps": ini_apps,
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

                review_path = instance_dir / "outputs" / "review.json"
                if not review_path.exists() or not _load_json(review_path).get(
                    "valid"
                ):
                    instance_errors.append(
                        "current Instance review is missing or invalid"
                    )
                noise_path = instance_dir / "outputs" / "noise.json"
                if not noise_path.exists():
                    instance_errors.append("outputs/noise.json is missing")
                else:
                    noise = _load_json(noise_path)
                    noise_conflicts = find_noise_structure_conflicts(noise)
                    if noise_conflicts:
                        instance_errors.extend(
                            f"Noise: {conflict}" for conflict in noise_conflicts
                        )
                    noise_command_count += len(noise.get("noise_commands", []))

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

    validator_rejected = reviewer_rejected = 0
    for log_dir in domains_dir.glob(
        "*/instances/*/scenarios/logs/*_failed_*"
    ):
        validation_path = log_dir / "validation_output.json"
        review_path = log_dir / "review_output.json"
        if validation_path.exists() and not _load_json(validation_path).get("valid"):
            validator_rejected += 1
        elif review_path.exists() and not _load_json(review_path).get("valid"):
            reviewer_rejected += 1
    generated = len(all_records) + validator_rejected + reviewer_rejected
    manifest = {
        "benchmark_id": plan["benchmark_id"],
        "valid": not errors,
        "summary": {
            "domain_count": len(actual_domains),
            "instance_count": sum(
                len(item["instances"]) for item in domain_entries
            ),
            "scenario_count": len(all_records),
            "event_count": sum(
                count * frequency for count, frequency in event_counts.items()
            ),
            "candidate_success": {
                "generated": generated,
                "accepted": len(all_records),
                "validator_rejected": validator_rejected,
                "reviewer_rejected": reviewer_rejected,
                "success_rate": (
                    round(len(all_records) / generated, 4) if generated else None
                ),
            },
            "noise_command_count": noise_command_count,
            "event_count_distribution": dict(sorted(event_counts.items())),
            "initial_data_app_counts": dict(sorted(ini_app_counts.items())),
        },
        "review_report": {"errors": errors, "warnings": warnings},
        "domains": domain_entries,
    }
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
    _create_instance(
        domain_spec,
        instance_spec,
        plan["benchmark_year"],
        stop_event,
    )
    instance_dir = DOMAINS_DIR / domain_id / "instances" / instance_id
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
    _configure_logging()
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
    _prepare_domains(selected_domains)

    jobs = []
    known_instances = set()
    for domain_spec in selected_domains:
        for instance_spec in instance_schedule(plan, domain_spec):
            known_instances.add(instance_spec["instance_id"])
            if instance_ids is None or instance_spec["instance_id"] in instance_ids:
                jobs.append((domain_spec, instance_spec))
    unknown_instances = set(instance_ids or []) - known_instances
    if unknown_instances:
        raise ValueError(f"unknown Instance ids: {sorted(unknown_instances)}")

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results, failures = [], []
    stop_event = Event()
    for domain_spec, instance_spec in jobs:
        domain_id = domain_spec["domain_id"]
        instance_id = instance_spec["instance_id"]
        try:
            results.append(
                _build_one_instance(plan, domain_spec, instance_spec, stop_event)
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
            LOGGER.info(f"[{domain_id}/{instance_id}] BUILD FAILED: {failure['error']}")

    run_elapsed = round(time.monotonic() - started, 2)
    report = {
        "complete": not failures
        and all(item["status"] == "complete" for item in results),
        "run_elapsed_seconds": run_elapsed,
        "construction_elapsed_seconds": sum(
            item["elapsed_seconds"] for item in results
        ),
        "selected_instance_count": len(jobs),
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
            domain_ids=[spec["domain_id"] for spec in selected_domains]
        )
    else:
        audit = None
    return report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", dest="domain_ids")
    parser.add_argument("--instance", action="append", dest="instance_ids")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="audit existing data without running construction",
    )
    args = parser.parse_args()
    if args.audit_only:
        if args.instance_ids:
            parser.error("--audit-only does not accept --instance")
        raise SystemExit(
            0 if audit_benchmark(domain_ids=args.domain_ids)["valid"] else 1
        )
    report, audit = build_benchmark(
        domain_ids=args.domain_ids,
        instance_ids=args.instance_ids,
    )
    valid = report["complete"] and (audit is None or audit["valid"])
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
