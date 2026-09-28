"""Generate one proactive Scenario inside an Instance."""

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

from env.apps import Database, Files
from llm_client import LLMClient

from .prompt import TRIGGER_DEFINITIONS, build_scenario_creator_prompt


ROOT_DIR = Path(__file__).resolve().parents[1]
DOMAINS_DIR = ROOT_DIR / "domains"


class ScenarioOutputError(ValueError):
    """The Creator response was not a usable JSON object."""


def next_reference_date(instance_dir, year):
    """Choose the next date from a random, year-spanning ordered sample."""
    accepted = load_accepted_scenarios(instance_dir)
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)
    earliest = (
        max(date.fromisoformat(item["reference_time"]) for item in accepted)
        + timedelta(days=1)
        if accepted
        else year_start
    )
    remaining = len(TRIGGER_DEFINITIONS) * 2 - len(accepted)
    available_days = (year_end - earliest).days + 1
    if available_days < remaining:
        raise ValueError("No chronological reference date remains in benchmark year")
    offset = min(random.sample(range(available_days), remaining))
    return (earliest + timedelta(days=offset)).isoformat()


def get_instance_dir(domain_id, instance_id):
    return DOMAINS_DIR / domain_id / "instances" / instance_id


def load_instance_profile(instance_dir):
    return json.loads(
        (Path(instance_dir) / "profile.json").read_text(encoding="utf-8")
    )


def load_instance_state(instance_dir, instance_id):
    """Read shared database, Files, and native App state."""
    instance_dir = Path(instance_dir)
    database = Database(instance_dir / "data" / f"{instance_id}.sqlite")
    tables = database.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )["rows"]
    output_path = instance_dir / "outputs" / "instance.json"
    instance_output = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists() else {"commands": []}
    )
    file_descriptions = {
        command["parameter"]["name"]: command["description"]
        for command in instance_output["commands"]
        if command["app"] == "files"
    }
    files = Files(instance_dir / "data")
    apps = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((instance_dir / "app").glob("*.json"))
    }
    return {
        "database_schema": [sql for _, sql in tables],
        "database_rows": {
            name: database.execute(f'SELECT * FROM "{name}"') for name, _ in tables
        },
        "files": [
            {
                "name": item["name"],
                "description": file_descriptions.get(
                    item["name"],
                    "Text file in the shared Instance state.",
                ),
                "content": files.get_file(item["name"])["content"],
            }
            for item in files.list_files()
        ],
        "apps": apps,
    }


def published_scenario(original, public):
    """Overlay published content without rewriting historical construction logs."""
    scenario = {**original, **public}
    events = public.get("events", {})
    if isinstance(events, dict) and events:
        first = next(iter(events.values()))
        try:
            scenario["reference_time"] = datetime.fromisoformat(
                first["time"].replace("Z", "+00:00")
            ).date().isoformat()
        except (KeyError, TypeError, ValueError, AttributeError):
            pass  # Structural validation reports malformed event timestamps.
    return scenario


def load_published_scenario(public_path):
    """Add only validation metadata to an already materialized public Scenario."""
    public_path = Path(public_path)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    if not isinstance(public, dict) or set(public) != {
        "scenario_description", "events"
    }:
        raise ValueError(
            "Published Scenario must contain only scenario_description and events"
        )
    return published_scenario(
        {
            "scenario_id": public_path.stem,
            "trigger_type": public_path.parent.name,
            "reference_time": None,
            "ini_data_commands": [],
        },
        public,
    )


def load_accepted_scenarios(instance_dir):
    """Read current events, with construction metadata only when still present."""
    instance_dir = Path(instance_dir)
    scenarios = []
    for trigger in TRIGGER_DEFINITIONS:
        trigger_dir = instance_dir / "scenarios" / trigger
        for public_path in sorted(trigger_dir.glob(f"{trigger}_*.json")):
            hidden = (
                instance_dir / "scenarios" / "logs" / public_path.stem
                / "scenario_output.json"
            )
            scenario = load_published_scenario(public_path)
            if hidden.exists():
                scenario = published_scenario(
                    json.loads(hidden.read_text(encoding="utf-8")),
                    json.loads(public_path.read_text(encoding="utf-8")),
                )
            scenarios.append(scenario)
    return scenarios


def scenario_context_summary(scenario):
    """Keep the causal design signals needed for set-level prompt context."""
    return {
        "scenario_id": scenario["scenario_id"],
        "trigger_type": scenario["trigger_type"],
        "scenario_description": scenario["scenario_description"],
        "event_sequence": [
            f"{event['app']}.{event['function']}"
            for event in scenario["events"].values()
        ],
        "initialization_sources": sorted(
            {
                f"{command['app']}.{command['function']}"
                for command in scenario["ini_data_commands"]
            }
        ),
    }


def load_scenario_context_summaries(instance_dir):
    instance_dir = Path(instance_dir)
    return [
        {"instance_id": instance_dir.name, **scenario_context_summary(scenario)}
        for scenario in load_accepted_scenarios(instance_dir)
    ]


def load_domain_scenario_context_summaries(domain_dir):
    return [
        summary
        for instance_dir in sorted((Path(domain_dir) / "instances").glob("*"))
        if instance_dir.is_dir()
        for summary in load_scenario_context_summaries(instance_dir)
    ]


class ScenarioCreator:
    """Create one trigger-specific Scenario with one LLM request."""

    def __init__(self):
        self.client = LLMClient(temperature=0.7)

    def create_scenario(
        self,
        domain_id,
        instance_id,
        trigger_type,
        reference_date,
        scenario_id,
        retry_feedback,
    ):
        instance_dir = get_instance_dir(domain_id, instance_id)
        profile = load_instance_profile(instance_dir)
        accepted = load_domain_scenario_context_summaries(DOMAINS_DIR / domain_id)
        prompt = build_scenario_creator_prompt(
            trigger_type=trigger_type,
            scenario_id=scenario_id,
            instance_context={
                "overview": profile["instance_description"],
                **load_instance_state(instance_dir, instance_id),
            },
            personal_description=profile["personal_description"],
            same_trigger_scenarios=[
                item for item in accepted if item["trigger_type"] == trigger_type
            ],
            other_scenarios=[
                item for item in accepted if item["trigger_type"] != trigger_type
            ],
            reference_date=reference_date,
            retry_feedback=retry_feedback,
        )
        log_dir = instance_dir / "scenarios" / "logs" / scenario_id
        raw_output = self.client.generate_text(
            prompt,
            request_label=f"{domain_id}/{instance_id}/{scenario_id}/Creator",
        )
        try:
            generated = json.loads(raw_output)
        except json.JSONDecodeError as error:
            raise ScenarioOutputError(
                f"Creator returned invalid JSON: {error.msg}"
            ) from error
        if not isinstance(generated, dict):
            raise ScenarioOutputError("Creator must return one JSON object")
        scenario = {
            **generated,
            "scenario_id": scenario_id,
            "trigger_type": trigger_type,
            "reference_time": reference_date,
        }
        log_dir.mkdir(parents=True)
        (log_dir / "scenario_output.json").write_text(
            json.dumps(scenario, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return scenario
