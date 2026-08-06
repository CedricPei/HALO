"""Orchestrate one Scenario round and commit only an accepted result."""

import json
from pathlib import Path

from command_parser import CommandParser

from .llm_client import LLMTransportError
from .reviewer import Reviewer
from .scenario_creator import ScenarioCreator, get_instance_dir, random_reference_date
from .validator import validate_candidate


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _next_failed_id(logs_dir):
    numbers = [
        int(path.name.removeprefix("failed_"))
        for path in Path(logs_dir).glob("failed_*")
        if path.name.removeprefix("failed_").isdigit()
    ]
    return f"failed_{max(numbers, default=0) + 1}"


def _archive_attempt(instance_dir, trigger_type, scenario_id):
    logs_dir = Path(instance_dir) / "scenarios" / trigger_type / "logs"
    failed_id = _next_failed_id(logs_dir)
    target = logs_dir / failed_id
    (logs_dir / scenario_id).rename(target)
    path = target / "scenario_output.json"
    if path.exists():
        output = json.loads(path.read_text(encoding="utf-8"))
        output["scenario_id"] = failed_id
        _write_json(path, output)
    return failed_id


def _record_error(instance_dir, trigger_type, error, scenario_id=None):
    logs_dir = Path(instance_dir) / "scenarios" / trigger_type / "logs"
    source = logs_dir / scenario_id if scenario_id else None
    if source and source.exists():
        failed_id = _archive_attempt(instance_dir, trigger_type, scenario_id)
    else:
        failed_id = _next_failed_id(logs_dir)
        (logs_dir / failed_id).mkdir(parents=True)
    _write_json(
        logs_dir / failed_id / "error.json",
        {"error": f"{type(error).__name__}: {error}"},
    )
    return failed_id


def build_scenario(
    domain_id,
    instance_id,
    trigger_type,
    reference_date=None,
    max_attempts=5,
    max_transport_errors=6,
    scenario_creator=None,
    reviewer=None,
):
    """Retry the newest combined Scenario/Data output until both gates pass."""
    scenario_creator = scenario_creator or ScenarioCreator()
    reviewer = reviewer or Reviewer()
    instance_dir = get_instance_dir(
        scenario_creator.domains_dir, domain_id, instance_id
    )
    reference_date = reference_date or random_reference_date()
    rejected = []
    content_failures = transport_failures = attempts = 0

    while content_failures < max_attempts:
        attempts += 1
        scenario = None
        try:
            scenario = scenario_creator.create_scenario(
                domain_id,
                instance_id,
                trigger_type,
                reference_date,
                retry_feedback=rejected,
            )
            conflicts = validate_candidate(
                instance_dir,
                instance_id,
                scenario,
                expected_trigger_type=trigger_type,
            )
            validation = {"valid": not conflicts, "conflicts": conflicts}
            log_dir = (
                instance_dir / "scenarios" / trigger_type / "logs"
                / scenario["scenario_id"]
            )
            _write_json(log_dir / "validation_output.json", validation)
            review = None
            if validation["valid"]:
                review = reviewer.review(
                    domain_id, instance_id, trigger_type, scenario
                )
                _write_json(log_dir / "review_output.json", review)

            if validation["valid"] and review["valid"]:
                CommandParser.for_instance(
                    instance_dir, instance_id
                ).execute_commands({"commands": scenario["ini_data_commands"]})
                _write_json(instance_dir / "review_output.json", review)
                _write_json(
                    instance_dir / "scenarios" / trigger_type
                    / f"{scenario['scenario_id']}.json",
                    {
                        "scenario_description": scenario["scenario_description"],
                        "events": scenario["events"],
                    },
                )
                return {
                    "scenario": scenario,
                    "validation": validation,
                    "review": review,
                    "attempts": attempts,
                    "transport_failures": transport_failures,
                    "rejected": rejected,
                }
        except Exception as error:
            failed_id = _record_error(
                instance_dir,
                trigger_type,
                error,
                scenario["scenario_id"] if scenario else None,
            )
            rejected.append(
                {"log_id": failed_id, "error": f"{type(error).__name__}: {error}"}
            )
            if isinstance(error, LLMTransportError):
                transport_failures += 1
                if transport_failures >= max_transport_errors:
                    break
            else:
                content_failures += 1
            continue

        issues = validation["conflicts"] if not validation["valid"] else review["issues"]
        failed_id = _archive_attempt(instance_dir, trigger_type, scenario["scenario_id"])
        rejected.append({"log_id": failed_id, "issues": issues})
        content_failures += 1

    raise RuntimeError(
        f"Scenario creation failed after {content_failures} content and "
        f"{transport_failures} transport failures: "
        + json.dumps(rejected, ensure_ascii=False)
    )
