"""Deterministically validate Scenario and Noise candidates."""

import json
import re
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path

from command_parser import CommandParser

from .prompt import TRIGGER_DEFINITIONS
from .scenario_creator import load_accepted_scenarios


RETRIEVAL_APPS = {"database", "files"}
MIN_EVENT_COUNT = 4
MAX_EVENT_COUNT = 6
MIN_NON_RETRIEVAL_EVENT_APPS = 3


def _timestamp(value, label, conflicts):
    if not isinstance(value, str):
        conflicts.append(f"{label} must be an ISO 8601 timestamp with timezone")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        conflicts.append(f"{label} must be an ISO 8601 timestamp with timezone")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        conflicts.append(f"{label} must include a timezone")
        return None
    return parsed


def _execute_checked(parser, command):
    result = parser.execute_command(command)
    if command["app"] == "database" and result.get("affected_rows") == 0:
        parameter = command["parameter"]
        sql = parameter["sql"] if isinstance(parameter, dict) else parameter
        if sql.lstrip().split(None, 1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
            raise RuntimeError("database command affected zero rows")
    return result


def _command_shape_conflicts(commands, label):
    conflicts = []
    if not isinstance(commands, list):
        return [f"{label} must be a list"]
    for index, command in enumerate(commands, 1):
        if not isinstance(command, dict) or set(command) != {
            "app", "function", "parameter"
        }:
            conflicts.append(
                f"{label} command {index} must contain app, function, parameter"
            )
            continue
        if not all(
            isinstance(command[field], str) and command[field]
            for field in ("app", "function")
        ):
            conflicts.append(
                f"{label} command {index} needs non-empty app and function names"
            )
    return conflicts


def find_structure_conflicts(scenario, expected_trigger_type):
    """Return machine-checkable Scenario schema conflicts."""
    if not isinstance(scenario, dict):
        return ["Scenario output must be one JSON object"]
    conflicts = []
    fields = [
        "scenario_id",
        "trigger_type",
        "reference_time",
        "scenario_description",
        "events",
        "ini_data_commands",
    ]
    if set(scenario) != set(fields):
        conflicts.append(f"Scenario fields must be exactly {fields}")

    trigger_type = scenario.get("trigger_type")
    if trigger_type not in TRIGGER_DEFINITIONS:
        conflicts.append(f"unknown trigger_type {trigger_type!r}")
    if trigger_type != expected_trigger_type:
        conflicts.append(
            f"trigger_type must be {expected_trigger_type!r}, got {trigger_type!r}"
        )
    reference_date = None
    try:
        reference_date = date.fromisoformat(scenario.get("reference_time"))
    except (TypeError, ValueError):
        conflicts.append("reference_time must be an ISO 8601 date")
    description = scenario.get("scenario_description")
    if not isinstance(description, str) or len(description.split()) < 20:
        conflicts.append("scenario_description is missing or too short")

    events = scenario.get("events")
    if (
        not isinstance(events, dict)
        or not MIN_EVENT_COUNT <= len(events) <= MAX_EVENT_COUNT
    ):
        conflicts.append(
            f"events must contain {MIN_EVENT_COUNT} to {MAX_EVENT_COUNT} checkpoints"
        )
        events = {}
    expected_keys = [f"event{index}" for index in range(1, len(events) + 1)]
    if list(events) != expected_keys:
        conflicts.append(f"event keys must be sequential: {expected_keys}")
    event_apps = set()
    first_event_time = previous_time = None
    for event_id, event in events.items():
        if not isinstance(event, dict) or set(event) != {
            "time", "app", "function", "parameter"
        }:
            conflicts.append(
                f"{event_id} must contain exactly time, app, function, parameter"
            )
            continue
        event_time = _timestamp(event.get("time"), f"{event_id} time", conflicts)
        if event_time is not None:
            if first_event_time is None:
                first_event_time = event_time
            if reference_date is not None and event_time.date() < reference_date:
                conflicts.append(f"{event_id} time precedes reference_time")
            if previous_time is not None and event_time <= previous_time:
                conflicts.append("Event times must be strictly increasing")
            previous_time = event_time
        app, function = event.get("app"), event.get("function")
        if (
            not isinstance(app, str)
            or not app
            or not isinstance(function, str)
            or not function
        ):
            conflicts.append(f"{event_id} needs non-empty app and function names")
            continue
        event_apps.add(app)
        parameter = event.get("parameter")
        record_time = (
            _timestamp(
                parameter["timestamp"],
                f"{event_id} parameter.timestamp",
                conflicts,
            )
            if isinstance(parameter, dict) and "timestamp" in parameter
            else None
        )
        if record_time is not None:
            if event_time is not None and record_time > event_time:
                conflicts.append(
                    f"{event_id} parameter.timestamp cannot follow Event time"
                )

    if (
        reference_date is not None
        and first_event_time is not None
        and first_event_time.date() != reference_date
    ):
        conflicts.append("event1 date must equal reference_time")

    non_retrieval_apps = sorted(event_apps - RETRIEVAL_APPS)
    if len(non_retrieval_apps) < MIN_NON_RETRIEVAL_EVENT_APPS:
        conflicts.append(
            f"events must use at least {MIN_NON_RETRIEVAL_EVENT_APPS} "
            "distinct non-retrieval Apps; "
            f"got {non_retrieval_apps}"
        )

    initialization = scenario.get("ini_data_commands")
    conflicts.extend(_command_shape_conflicts(initialization, "ini_data_commands"))
    commands = initialization if isinstance(initialization, list) else ()
    for index, command in enumerate(commands, 1):
        parameter = command.get("parameter") if isinstance(command, dict) else None
        if not isinstance(parameter, dict) or "timestamp" not in parameter:
            continue
        initialization_time = _timestamp(
            parameter["timestamp"],
            f"ini_data_commands command {index} parameter.timestamp",
            conflicts,
        )
        if initialization_time is None:
            continue
        if reference_date is not None and initialization_time.date() > reference_date:
            conflicts.append(
                f"ini_data_commands command {index} timestamp follows reference_time"
            )
        if first_event_time is not None and initialization_time > first_event_time:
            conflicts.append(
                f"ini_data_commands command {index} timestamp follows event1 time"
            )
    return conflicts


def _render(command):
    rendered = json.dumps(command, ensure_ascii=False)
    return rendered if len(rendered) <= 1000 else rendered[:1000] + "..."


def _copy_state(source, target):
    target.mkdir()
    shutil.copytree(Path(source) / "data", target / "data")
    shutil.copytree(Path(source) / "app", target / "app")


def _execute_events(instance_dir, instance_id, scenario, label):
    parser = CommandParser.for_instance(instance_dir, instance_id)
    for event_id, event in scenario["events"].items():
        try:
            _execute_checked(
                parser,
                {key: event[key] for key in ("app", "function", "parameter")},
            )
        except Exception as error:
            return [f"{label} {event_id} {_render(event)}: {error}"]
    return []


def find_execution_conflicts(instance_dir, instance_id, scenario):
    """Apply proposed initial data, then replay every Event branch in isolation."""
    with tempfile.TemporaryDirectory() as temporary:
        proposed = Path(temporary) / "proposed"
        _copy_state(instance_dir, proposed)
        parser = CommandParser.for_instance(proposed, instance_id)
        for index, command in enumerate(scenario["ini_data_commands"], 1):
            try:
                _execute_checked(parser, command)
            except Exception as error:
                return [f"ini_data command {index} {_render(command)}: {error}"]

        branches = [("candidate", scenario)] + [
            (item["scenario_id"], item)
            for item in load_accepted_scenarios(instance_dir)
        ]
        for index, (label, branch) in enumerate(branches):
            branch_dir = Path(temporary) / f"branch_{index}"
            _copy_state(proposed, branch_dir)
            conflicts = _execute_events(branch_dir, instance_id, branch, label)
            if conflicts:
                return conflicts
    return []


def validate_candidate(
    instance_dir,
    instance_id,
    scenario,
    expected_trigger_type,
):
    conflicts = find_structure_conflicts(scenario, expected_trigger_type)
    if conflicts:
        return conflicts
    return find_execution_conflicts(instance_dir, instance_id, scenario)


def validate_published_scenario(
    instance_dir,
    instance_id,
    scenario,
    expected_trigger_type,
):
    """Validate one accepted Scenario against the final shared state."""
    conflicts = find_structure_conflicts(scenario, expected_trigger_type)
    if conflicts:
        return conflicts
    with tempfile.TemporaryDirectory() as temporary:
        branch = Path(temporary) / "branch"
        _copy_state(instance_dir, branch)
        return _execute_events(
            branch, instance_id, scenario, scenario["scenario_id"]
        )


def find_noise_structure_conflicts(noise):
    """Return conflicts that can be checked without replaying Noise."""
    if not isinstance(noise, dict) or set(noise) != {"noise_commands"}:
        return ["Noise output must contain exactly noise_commands"]
    commands = noise.get("noise_commands")
    conflicts = _command_shape_conflicts(commands, "noise")
    if conflicts:
        return conflicts
    if not 12 <= len(commands) <= 20:
        return ["noise_commands must contain 12 to 20 commands"]
    for index, command in enumerate(commands, 1):
        if command["app"] == "database":
            parameter = command["parameter"]
            sql = parameter.get("sql", "") if isinstance(parameter, dict) else parameter
            if not isinstance(sql, str) or not sql.lstrip().upper().startswith("INSERT"):
                return [f"noise command {index} database SQL must be INSERT"]
    return []


def validate_noise(instance_dir, instance_id, noise):
    """Execute Noise in a copy and replay all accepted branches."""
    conflicts = find_noise_structure_conflicts(noise)
    if conflicts:
        return conflicts
    commands = noise["noise_commands"]

    with tempfile.TemporaryDirectory() as temporary:
        proposed = Path(temporary) / "noise"
        _copy_state(instance_dir, proposed)
        parser = CommandParser.for_instance(proposed, instance_id)
        for index, command in enumerate(commands, 1):
            try:
                _execute_checked(parser, command)
            except Exception as error:
                return [f"noise command {index} {_render(command)}: {error}"]
        for index, scenario in enumerate(load_accepted_scenarios(instance_dir)):
            branch = Path(temporary) / f"branch_{index}"
            _copy_state(proposed, branch)
            conflicts = _execute_events(
                branch, instance_id, scenario, scenario["scenario_id"]
            )
            if conflicts:
                return conflicts
    return []


def description_similarity(left, right):
    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "in", "for", "is",
        "this", "because", "scenario", "protagonist", "user",
    }
    words = lambda text: {
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }
    left_words, right_words = words(left), words(right)
    union = left_words | right_words
    return len(left_words & right_words) / len(union) if union else 1.0
