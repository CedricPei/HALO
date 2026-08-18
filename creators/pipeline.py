"""Orchestrate one Scenario round and commit only an accepted result."""

import json
import logging
import shutil
from concurrent.futures import CancelledError

from command_parser import CommandParser
from llm_client import LLMTransportError

from .reviewer import Reviewer, ReviewerOutputError
from .scenario_creator import (
    ScenarioCreator,
    ScenarioOutputError,
    get_instance_dir,
    next_reference_date,
)
from .validator import validate_candidate


LOGGER = logging.getLogger(__name__)


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _save_retry_feedback(path, feedback):
    path.parent.mkdir(exist_ok=True)
    _write_json(path, feedback)


def _next_failed_id(logs_dir, trigger_type):
    prefix = f"{trigger_type}_failed_"
    numbers = [
        int(path.name.removeprefix(prefix))
        for path in logs_dir.glob(f"{prefix}*")
        if path.name.removeprefix(prefix).isdigit()
    ]
    return f"{prefix}{max(numbers, default=0) + 1}"


def _archive_attempt(instance_dir, trigger_type, scenario_id):
    logs_dir = instance_dir / "scenarios" / "logs"
    failed_id = _next_failed_id(logs_dir, trigger_type)
    target = logs_dir / failed_id
    (logs_dir / scenario_id).rename(target)
    path = target / "scenario_output.json"
    output = json.loads(path.read_text(encoding="utf-8"))
    output["scenario_id"] = failed_id
    _write_json(path, output)
    return failed_id


def _discard_active_attempt(instance_dir, scenario_id):
    path = instance_dir / "scenarios" / "logs" / scenario_id
    shutil.rmtree(path, ignore_errors=True)


def _stop_if_requested(stop_event):
    if stop_event.is_set():
        raise CancelledError


def _candidate_summary(scenario):
    events = scenario.get("events") or {}
    event_values = events.values() if isinstance(events, dict) else ()
    return {
        "scenario_description": scenario.get("scenario_description"),
        "event_sequence": [
            f"{event.get('app')}.{event.get('function')}"
            for event in event_values
            if isinstance(event, dict)
        ],
    }


def build_scenario(
    domain_id,
    instance_id,
    trigger_type,
    scenario_number,
    benchmark_year,
    stop_event,
):
    """Try up to five revised candidates and commit the first accepted one.

    The LLM client retries transport failures in place. Malformed Reviewer
    output has a separate output-attempt budget against the same candidate.
    Neither failure becomes Creator semantic feedback.
    """
    scenario_creator, reviewer = ScenarioCreator(), Reviewer()
    instance_dir = get_instance_dir(domain_id, instance_id)
    reference_date = next_reference_date(instance_dir, benchmark_year)
    scenario_id = f"{trigger_type}_{scenario_number}"
    rejected = []
    feedback_path = (
        instance_dir / "scenarios" / "logs" / "retry" / f"{scenario_id}.json"
    )
    retry_feedback = (
        json.loads(feedback_path.read_text(encoding="utf-8"))
        if feedback_path.exists()
        else []
    )
    context = f"{domain_id}/{instance_id}/{scenario_id}"
    if retry_feedback:
        LOGGER.info("[%s] loaded %s persisted retry item(s)", context, len(retry_feedback))
    LOGGER.info(
        "[%s] Scenario started trigger=%s candidates=%s",
        context,
        trigger_type,
        5,
    )

    for attempt in range(1, 6):
        _stop_if_requested(stop_event)
        stage = "Creator"
        LOGGER.info(
            "[%s] attempt %s/%s Creator request started",
            context,
            attempt,
            5,
        )
        try:
            scenario = scenario_creator.create_scenario(
                domain_id,
                instance_id,
                trigger_type,
                reference_date,
                scenario_id,
                retry_feedback=retry_feedback,
            )
            _stop_if_requested(stop_event)
            stage = "Validator"
            conflicts = validate_candidate(
                instance_dir,
                instance_id,
                scenario,
                expected_trigger_type=trigger_type,
            )
            validation = {"valid": not conflicts, "conflicts": conflicts}
            log_dir = (
                instance_dir / "scenarios" / "logs" / scenario["scenario_id"]
            )
            _write_json(log_dir / "validation_output.json", validation)
            review = None
            if validation["valid"]:
                stage = "Reviewer"
                LOGGER.info(
                    "[%s] attempt %s/%s Reviewer request started event_count=%s",
                    context,
                    attempt,
                    5,
                    len(scenario["events"]),
                )
                for output_attempt in range(1, 6):
                    _stop_if_requested(stop_event)
                    try:
                        review = reviewer.review(
                            domain_id, instance_id, trigger_type, scenario
                        )
                        break
                    except ReviewerOutputError as error:
                        exhausted = output_attempt == 5
                        LOGGER.warning(
                            "[%s] attempt %s/%s Reviewer output %s/%s "
                            "invalid next=%s error=%s",
                            context,
                            attempt,
                            5,
                            output_attempt,
                            5,
                            "stop" if exhausted else "retry same candidate",
                            error,
                        )
                if review is None:
                    _discard_active_attempt(instance_dir, scenario_id)
                    LOGGER.error(
                        "[%s] Reviewer output remained unusable after 5 attempts; "
                        "Scenario has no semantic outcome",
                        context,
                    )
                    return {
                        "status": "unresolved",
                        "scenario_id": scenario_id,
                        "scenario": None,
                        "validation": validation,
                        "review": None,
                        "attempts": attempt,
                        "rejected": rejected,
                    }
                _stop_if_requested(stop_event)
                _write_json(log_dir / "review_output.json", review)

            if validation["valid"] and review["valid"]:
                _stop_if_requested(stop_event)
                stage = "Commit"
                CommandParser.for_instance(instance_dir, instance_id).execute_commands(
                    {"commands": scenario["ini_data_commands"]}
                )
                _write_json(instance_dir / "outputs" / "review.json", review)
                _write_json(
                    instance_dir
                    / "scenarios"
                    / trigger_type
                    / f"{scenario['scenario_id']}.json",
                    {
                        "scenario_description": scenario["scenario_description"],
                        "events": scenario["events"],
                    },
                )
                feedback_path.unlink(missing_ok=True)
                LOGGER.info(
                    "[%s] accepted on attempt %s/%s event_count=%s",
                    context,
                    attempt,
                    5,
                    len(scenario["events"]),
                )
                return {
                    "status": "accepted",
                    "scenario_id": scenario_id,
                    "scenario": scenario,
                    "validation": validation,
                    "review": review,
                    "attempts": attempt,
                    "rejected": rejected,
                }
        except CancelledError:
            _discard_active_attempt(instance_dir, scenario_id)
            raise
        except LLMTransportError as error:
            stop_event.set()
            _discard_active_attempt(instance_dir, scenario_id)
            LOGGER.error(
                "[%s] %s LLM requests exhausted; aborting benchmark: %s",
                context,
                stage,
                error,
            )
            raise
        except ScenarioOutputError as error:
            retry_feedback.append({"output_error": str(error)})
            _save_retry_feedback(feedback_path, retry_feedback)
            LOGGER.warning(
                "[%s] attempt %s/%s Creator output unusable retry=%s error=%s",
                context,
                attempt,
                5,
                attempt < 5,
                error,
            )
            continue
        except Exception:
            _discard_active_attempt(instance_dir, scenario_id)
            raise

        issues = (
            validation["conflicts"] if not validation["valid"] else review["issues"]
        )
        failed_id = _archive_attempt(
            instance_dir, trigger_type, scenario["scenario_id"]
        )
        rejected_by = "Validator" if not validation["valid"] else "Reviewer"
        rejected.append(
            {
                "log_id": failed_id,
                "rejected_by": rejected_by,
                "issues": issues,
                "discarded_candidate": _candidate_summary(scenario),
            }
        )
        retry_feedback.append(rejected[-1])
        _save_retry_feedback(feedback_path, retry_feedback)
        LOGGER.warning(
            "[%s] attempt %s/%s rejected_by=%s log_id=%s retry=%s issues=%s",
            context,
            attempt,
            5,
            rejected_by,
            failed_id,
            attempt < 5,
            issues,
        )

    LOGGER.error("[%s] unresolved after 5 Creator outputs", context)
    return {
        "status": "unresolved",
        "scenario_id": scenario_id,
        "scenario": None,
        "validation": None,
        "review": None,
        "attempts": attempt,
        "rejected": rejected,
    }
