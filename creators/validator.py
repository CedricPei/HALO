"""Deterministically validate Scenario and Noise candidates."""

import json
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path

from benchmark_plan import TRIGGER_TYPES
from command_parser import CommandParser


EVENT_FUNCTIONS = {
    "database": {"execute"},
    "files": {"create_file", "append_file", "replace_file", "delete_file"},
    "email": {"create_email", "delete_email", "create_draft", "update_draft"},
    "calendar": {
        "create_event", "update_event", "delete_event",
        "create_reminder", "update_reminder", "delete_reminder",
    },
    "messaging": {
        "create_conversation", "create_message", "update_message",
        "create_draft", "update_draft",
    },
    "contacts": {"create_contact", "update_contact", "delete_contact"},
    "wallet": {
        "update_wallet", "add_funds", "pay", "update_transaction_status",
        "refund_transaction", "create_draft", "update_draft",
    },
}

NOISE_FUNCTIONS = {
    "database": {"execute"},
    "files": {"create_file", "append_file", "replace_file"},
    "email": {"create_email"},
    "calendar": {"create_event", "create_reminder"},
    "messaging": {"create_conversation", "create_message"},
    "contacts": {"create_contact"},
    "wallet": {"add_funds", "pay"},
}


def _execute_checked(parser, command):
    result = parser.execute_command(command)
    if command["app"] == "database" and result.get("affected_rows") == 0:
        parameter = command["parameter"]
        sql = parameter["sql"] if isinstance(parameter, dict) else parameter
        if sql.lstrip().split(None, 1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
            raise RuntimeError("database command affected zero rows")
    return result


def _command_shape_conflicts(commands, label, functions):
    conflicts = []
    if not isinstance(commands, list):
        return [f"{label} must be a list"]
    for index, command in enumerate(commands, 1):
        if not isinstance(command, dict) or list(command) != [
            "app", "function", "parameter"
        ]:
            conflicts.append(
                f"{label} command {index} must contain app, function, parameter"
            )
            continue
        app, function = command.get("app"), command.get("function")
        if app not in functions or function not in functions[app]:
            conflicts.append(
                f"{label} command {index} uses unsupported {app}.{function}"
            )
    return conflicts


def find_structure_conflicts(scenario, expected_trigger_type=None):
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
    if list(scenario) != fields:
        conflicts.append(f"Scenario fields must be exactly {fields} in order")

    trigger_type = scenario.get("trigger_type")
    if trigger_type not in TRIGGER_TYPES:
        conflicts.append(f"unknown trigger_type {trigger_type!r}")
    if expected_trigger_type and trigger_type != expected_trigger_type:
        conflicts.append(
            f"trigger_type must be {expected_trigger_type!r}, got {trigger_type!r}"
        )
    try:
        date.fromisoformat(scenario.get("reference_time"))
    except (TypeError, ValueError):
        conflicts.append("reference_time must be an ISO 8601 date")
    description = scenario.get("scenario_description")
    if not isinstance(description, str) or len(description.split()) < 20:
        conflicts.append("scenario_description is missing or too short")

    events = scenario.get("events")
    if not isinstance(events, dict) or not 3 <= len(events) <= 6:
        conflicts.append("events must contain 3 to 6 checkpoints")
        events = {}
    expected_keys = [f"event{index}" for index in range(1, len(events) + 1)]
    if list(events) != expected_keys:
        conflicts.append(f"event keys must be sequential: {expected_keys}")
    for event_id, event in events.items():
        if not isinstance(event, dict) or list(event) != [
            "app", "function", "parameter"
        ]:
            conflicts.append(
                f"{event_id} must contain exactly app, function, parameter"
            )
            continue
        app, function = event.get("app"), event.get("function")
        if app not in EVENT_FUNCTIONS or function not in EVENT_FUNCTIONS[app]:
            conflicts.append(f"{event_id} uses unsupported {app}.{function}")

    conflicts.extend(
        _command_shape_conflicts(
            scenario.get("ini_data_commands"), "ini_data_commands", EVENT_FUNCTIONS
        )
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
            _execute_checked(parser, event)
        except Exception as error:
            return [f"{label} {event_id} {_render(event)}: {error}"]
    return []


def _accepted_scenarios(instance_dir):
    scenarios_dir = Path(instance_dir) / "scenarios"
    for trigger_type in TRIGGER_TYPES:
        for public_path in sorted(
            (scenarios_dir / trigger_type).glob(f"{trigger_type}_*.json")
        ):
            hidden = (
                scenarios_dir / trigger_type / "logs" / public_path.stem
                / "scenario_output.json"
            )
            yield json.loads(hidden.read_text(encoding="utf-8"))


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
            (item["scenario_id"], item) for item in _accepted_scenarios(instance_dir)
        ]
        for index, (label, branch) in enumerate(branches):
            branch_dir = Path(temporary) / f"branch_{index}"
            _copy_state(proposed, branch_dir)
            conflicts = _execute_events(branch_dir, instance_id, branch, label)
            if conflicts:
                return conflicts
    return []


def find_published_execution_conflicts(instance_dir, instance_id, scenario):
    """Replay one accepted Event branch from the final shared initial state."""
    with tempfile.TemporaryDirectory() as temporary:
        branch = Path(temporary) / "branch"
        _copy_state(instance_dir, branch)
        return _execute_events(branch, instance_id, scenario, scenario["scenario_id"])


def validate_candidate(
    instance_dir,
    instance_id,
    scenario,
    expected_trigger_type=None,
    published=False,
):
    conflicts = find_structure_conflicts(scenario, expected_trigger_type)
    if conflicts:
        return conflicts
    if published:
        return find_published_execution_conflicts(instance_dir, instance_id, scenario)
    return find_execution_conflicts(instance_dir, instance_id, scenario)


def validate_noise(instance_dir, instance_id, noise):
    """Execute Noise in a copy and replay all accepted branches."""
    if not isinstance(noise, dict) or list(noise) != ["noise_commands"]:
        return ["Noise output must contain exactly noise_commands"]
    commands = noise.get("noise_commands")
    conflicts = _command_shape_conflicts(commands, "noise", NOISE_FUNCTIONS)
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

    with tempfile.TemporaryDirectory() as temporary:
        proposed = Path(temporary) / "noise"
        _copy_state(instance_dir, proposed)
        parser = CommandParser.for_instance(proposed, instance_id)
        for index, command in enumerate(commands, 1):
            try:
                _execute_checked(parser, command)
            except Exception as error:
                return [f"noise command {index} {_render(command)}: {error}"]
        for index, scenario in enumerate(_accepted_scenarios(instance_dir)):
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
