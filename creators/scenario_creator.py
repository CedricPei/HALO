"""Generate one proactive Scenario inside an Instance."""

import json
import random
from datetime import date, timedelta
from pathlib import Path

from benchmark_plan import TRIGGER_TYPES, load_plan
from env.apps import Database, Files

from .llm_client import LLMClient
from .prompt import build_scenario_creator_prompt


ROOT_DIR = Path(__file__).resolve().parents[1]
DOMAINS_DIR = ROOT_DIR / "domains"
APPS_REFERENCE_PATH = ROOT_DIR / "env" / "apps.md"


def random_reference_date():
    year = load_plan()["design"]["benchmark_year"]
    start = date(year, 1, 1)
    day_count = (date(year + 1, 1, 1) - start).days
    return (start + timedelta(days=random.randrange(day_count))).isoformat()


def get_instance_dir(domains_dir, domain_id, instance_id):
    return Path(domains_dir) / domain_id / "instances" / instance_id


def load_instance_context(instance_dir, instance_id):
    """Read shared database, Files, descriptions, and Wallet state."""
    instance_dir = Path(instance_dir)
    database = Database(instance_dir / "data" / f"{instance_id}.sqlite")
    tables = database.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )["rows"]
    description = json.loads(
        (instance_dir / "description.json").read_text(encoding="utf-8")
    )
    file_descriptions = {
        item["name"]: item["description"]
        for item in description["file_description"]
    }
    files = Files(instance_dir / "data")
    apps = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((instance_dir / "app").glob("*.json"))
    }
    return {
        "description": description["instance_description"],
        "database_schema": [sql for _, sql in tables],
        "database_rows": {
            name: database.execute(f'SELECT * FROM "{name}"') for name, _ in tables
        },
        "files": [
            {
                "name": item["name"],
                "description": file_descriptions[item["name"]],
                "content": files.get_file(item["name"])["content"],
            }
            for item in files.list_files()
        ],
        "apps": apps,
    }


def load_personal_description(instance_dir):
    return json.loads(
        (Path(instance_dir) / "description.json").read_text(encoding="utf-8")
    )["personal_description"]


def load_accepted_scenarios(instance_dir):
    """Load every accepted hidden Scenario."""
    instance_dir = Path(instance_dir)
    scenarios = []
    for trigger in TRIGGER_TYPES:
        trigger_dir = instance_dir / "scenarios" / trigger
        for public_path in sorted(trigger_dir.glob(f"{trigger}_*.json")):
            hidden = trigger_dir / "logs" / public_path.stem / "scenario_output.json"
            scenarios.append(json.loads(hidden.read_text(encoding="utf-8")))
    return scenarios


def next_scenario_id(instance_dir, trigger_type):
    trigger_dir = Path(instance_dir) / "scenarios" / trigger_type
    numbers = [
        int(path.stem.removeprefix(f"{trigger_type}_"))
        for path in trigger_dir.glob(f"{trigger_type}_*.json")
        if path.stem.removeprefix(f"{trigger_type}_").isdigit()
    ]
    return f"{trigger_type}_{max(numbers, default=0) + 1}"


def assigned_event_count(scenario_id, trigger_type):
    if scenario_id.endswith("_1"):
        return 5
    return 4 if trigger_type in {"dependency", "risk"} else 6


class ScenarioCreator:
    """Create one trigger-specific Scenario with one LLM request."""

    def __init__(self, client=None, domains_dir=DOMAINS_DIR):
        self.client = client or LLMClient(temperature=0.7)
        self.domains_dir = Path(domains_dir)

    def create_scenario(
        self,
        domain_id,
        instance_id,
        trigger_type,
        reference_date=None,
        retry_feedback=None,
    ):
        instance_dir = get_instance_dir(self.domains_dir, domain_id, instance_id)
        scenario_id = next_scenario_id(instance_dir, trigger_type)
        event_count = assigned_event_count(scenario_id, trigger_type)
        accepted = load_accepted_scenarios(instance_dir)
        prompt = build_scenario_creator_prompt(
            trigger_type=trigger_type,
            scenario_id=scenario_id,
            instance_context=load_instance_context(instance_dir, instance_id),
            personal_description=load_personal_description(instance_dir),
            same_trigger_scenarios=[
                item for item in accepted if item["trigger_type"] == trigger_type
            ],
            other_scenarios=[
                item for item in accepted if item["trigger_type"] != trigger_type
            ],
            app_reference=APPS_REFERENCE_PATH.read_text(encoding="utf-8"),
            reference_date=reference_date or random_reference_date(),
            retry_feedback=retry_feedback,
            target_event_count=event_count,
        )
        try:
            generated = json.loads(self.client.generate_json(prompt))
        except json.JSONDecodeError as error:
            raise ValueError(f"LLM returned invalid JSON: {error.msg}") from error
        events = generated.get("events")
        if not isinstance(events, dict) or len(events) != event_count:
            actual = len(events) if isinstance(events, dict) else "invalid"
            raise ValueError(f"Scenario must contain {event_count} Events; got {actual}")
        scenario = {
            "scenario_id": scenario_id,
            "trigger_type": trigger_type,
            "reference_time": reference_date,
            **generated,
        }
        log_dir = (
            instance_dir / "scenarios" / trigger_type / "logs" / scenario_id
        )
        log_dir.mkdir(parents=True)
        (log_dir / "scenario_output.json").write_text(
            json.dumps(scenario, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return scenario


def create_scenario(domain_id, instance_id, trigger_type, reference_date=None, retry_feedback=None):
    return ScenarioCreator().create_scenario(
        domain_id, instance_id, trigger_type, reference_date, retry_feedback
    )
