"""Audit a complete benchmark and write one concise manifest.json."""

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from benchmark_plan import TRIGGER_TYPES, instance_schedule, load_plan
from creators.validator import description_similarity, validate_candidate


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DOMAINS = ROOT_DIR / "domains"
DEFAULT_OUTPUT = ROOT_DIR / "benchmark"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_benchmark(domains_dir=DEFAULT_DOMAINS, output_dir=DEFAULT_OUTPUT):
    """Validate the planned hierarchy and emit only manifest.json."""
    plan = load_plan()
    domains_dir, output_dir = Path(domains_dir), Path(output_dir)
    errors, warnings, domain_entries, all_records = [], [], [], []
    event_counts, ini_app_counts = Counter(), Counter()
    noise_command_count = 0

    expected_domains = {item["domain_id"] for item in plan["domains"]}
    actual_domains = {
        path.name for path in domains_dir.iterdir() if path.is_dir()
    } if domains_dir.exists() else set()
    if actual_domains != expected_domains:
        errors.append(
            f"Domain directories differ: missing={sorted(expected_domains-actual_domains)}, "
            f"unexpected={sorted(actual_domains-expected_domains)}"
        )

    for domain_spec in plan["domains"]:
        domain_id = domain_spec["domain_id"]
        domain_dir = domains_dir / domain_id
        domain_errors, instance_entries = [], []
        if not (domain_dir / "domain.json").exists():
            domain_errors.append("domain.json is missing")

        expected_instances = instance_schedule(plan, domain_spec)
        for instance_spec in expected_instances:
            instance_id = instance_spec["instance_id"]
            instance_dir = domain_dir / "instances" / instance_id
            instance_errors, scenario_entries, records = [], [], []
            trigger_counts = Counter()
            if not instance_dir.exists():
                instance_errors.append("Instance directory is missing")
            else:
                try:
                    output = _load(instance_dir / "instance_output.json")
                    if output.get("instance_id") != instance_id:
                        instance_errors.append("instance_output.json has wrong ID")
                    _load(instance_dir / "description.json")
                except Exception as error:
                    instance_errors.append(f"unreadable Instance metadata: {error}")

                for trigger in TRIGGER_TYPES:
                    trigger_dir = instance_dir / "scenarios" / trigger
                    public_paths = sorted(trigger_dir.glob(f"{trigger}_*.json"))
                    trigger_counts[trigger] = len(public_paths)
                    expected = plan["design"]["scenarios_per_trigger"]
                    if len(public_paths) != expected:
                        instance_errors.append(
                            f"{trigger} has {len(public_paths)} Scenarios; expected {expected}"
                        )
                    for public_path in public_paths:
                        scenario_id = public_path.stem
                        log_dir = trigger_dir / "logs" / scenario_id
                        required = [
                            "scenario_output.json",
                            "validation_output.json",
                            "review_output.json",
                        ]
                        missing = [name for name in required if not (log_dir / name).exists()]
                        if missing:
                            instance_errors.append(f"{scenario_id} missing {missing}")
                            continue
                        try:
                            scenario = _load(log_dir / "scenario_output.json")
                            validation = _load(log_dir / "validation_output.json")
                            review = _load(log_dir / "review_output.json")
                            public = _load(public_path)
                        except Exception as error:
                            instance_errors.append(f"{scenario_id} unreadable: {error}")
                            continue
                        if validation != {"valid": True, "conflicts": []}:
                            instance_errors.append(f"{scenario_id} failed Validator")
                        if not review.get("valid") or review.get("issues"):
                            instance_errors.append(f"{scenario_id} failed Reviewer")
                        if public != {
                            "scenario_description": scenario.get("scenario_description"),
                            "events": scenario.get("events"),
                        }:
                            instance_errors.append(f"{scenario_id} public output differs")
                        for conflict in validate_candidate(
                            instance_dir,
                            instance_id,
                            scenario,
                            expected_trigger_type=trigger,
                            published=True,
                        ):
                            instance_errors.append(f"{scenario_id}: {conflict}")

                        ini_apps = sorted(
                            {item["app"] for item in scenario["ini_data_commands"]}
                        )
                        for app in ini_apps:
                            ini_app_counts[app] += 1
                        count = len(scenario["events"])
                        event_counts[count] += 1
                        scenario_entries.append(
                            {
                                "scenario_id": scenario_id,
                                "trigger_type": trigger,
                                "event_count": count,
                                "ini_data_command_count": len(
                                    scenario["ini_data_commands"]
                                ),
                                "ini_data_apps": ini_apps,
                                "path": str(public_path.relative_to(ROOT_DIR)),
                                "sha256": _hash(public_path),
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

                review_path = instance_dir / "review_output.json"
                if not review_path.exists() or not _load(review_path).get("valid"):
                    instance_errors.append("current Instance review is missing or invalid")
                noise_output_path = instance_dir / "noise_output.json"
                noise_validation_path = instance_dir / "noise_validation.json"
                if not noise_output_path.exists() or not noise_validation_path.exists():
                    instance_errors.append("final Noise artifacts are missing")
                else:
                    noise = _load(noise_output_path)
                    noise_validation = _load(noise_validation_path)
                    if noise_validation != {"valid": True, "conflicts": []}:
                        instance_errors.append("final Noise validation is not valid")
                    else:
                        noise_command_count += len(noise.get("noise_commands", []))

            for left, right in combinations(records, 2):
                score = description_similarity(left["description"], right["description"])
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
                    "trigger_counts": {key: trigger_counts[key] for key in TRIGGER_TYPES},
                    "scenarios": scenario_entries,
                }
            )
            errors.extend(f"{domain_id}/{instance_id}: {item}" for item in instance_errors)

        domain_entries.append(
            {
                "domain_id": domain_id,
                "title": domain_spec["title"],
                "split": domain_spec["split"],
                "valid": not domain_errors and all(item["valid"] for item in instance_entries),
                "errors": domain_errors,
                "instances": instance_entries,
            }
        )
        errors.extend(f"{domain_id}: {item}" for item in domain_errors)

    rejected = len(list(domains_dir.glob("*/instances/*/scenarios/*/logs/failed_*")))
    manifest = {
        "benchmark_id": plan["benchmark_id"],
        "valid": not errors,
        "summary": {
            "domain_count": len(actual_domains),
            "instance_count": sum(len(item["instances"]) for item in domain_entries),
            "scenario_count": len(all_records),
            "event_count": sum(
                count * frequency for count, frequency in event_counts.items()
            ),
            "rejected_candidate_count": rejected,
            "noise_command_count": noise_command_count,
            "event_count_distribution": dict(sorted(event_counts.items())),
            "initial_data_app_counts": dict(sorted(ini_app_counts.items())),
        },
        "review_report": {"errors": errors, "warnings": warnings},
        "domains": domain_entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    raise SystemExit(0 if audit_benchmark()["valid"] else 1)


if __name__ == "__main__":
    main()
