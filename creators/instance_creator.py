"""Generate and initialize one concrete Instance inside a human Domain."""

import json
import re
import shutil
import tempfile
from pathlib import Path

from benchmark_plan import TRIGGER_TYPES, load_plan
from command_parser import CommandParser

from .llm_client import LLMClient
from .prompt import build_instance_creator_prompt


ROOT_DIR = Path(__file__).resolve().parents[1]
DOMAINS_DIR = ROOT_DIR / "domains"
APPS_REFERENCE_PATH = ROOT_DIR / "env" / "apps.md"


def find_instance_conflicts(instance, expected_instance_id=None):
    """Return deterministic format conflicts before executing commands."""
    if not isinstance(instance, dict):
        return ["Instance output must be one JSON object"]
    conflicts = []
    fields = [
        "instance_id",
        "instance_description",
        "personal_description",
        "commands",
    ]
    if list(instance) != fields:
        conflicts.append(f"top-level fields must be exactly {fields} in order")

    instance_id = instance.get("instance_id")
    if not isinstance(instance_id, str) or not re.fullmatch(
        r"[a-z][a-z0-9_]*", instance_id
    ):
        conflicts.append("instance_id must be lowercase snake_case")
    if expected_instance_id and instance_id != expected_instance_id:
        conflicts.append(f"instance_id must equal {expected_instance_id!r}")
    for field in ("instance_description", "personal_description"):
        if not isinstance(instance.get(field), str) or not instance[field].strip():
            conflicts.append(f"{field} must be a non-empty string")

    commands = instance.get("commands")
    if not isinstance(commands, list):
        return conflicts + ["commands must be a list"]

    schemas, files, profiles, wallets, groups = [], [], [], [], []
    for index, command in enumerate(commands, 1):
        if not isinstance(command, dict):
            conflicts.append(f"command {index} must be an object")
            continue
        app, function, parameter = (
            command.get("app"),
            command.get("function"),
            command.get("parameter"),
        )
        if app == "database" and function == "execute" and isinstance(parameter, str):
            verb = parameter.lstrip().split(None, 1)[0].upper() if parameter.strip() else ""
            if verb == "CREATE":
                schemas.append(command)
                groups.append("definition")
            elif verb == "INSERT":
                profiles.append(command)
                groups.append("initialization")
            else:
                conflicts.append(f"command {index} must be CREATE or profile INSERT")
        elif app == "files" and function == "create_file":
            files.append(command)
            groups.append("definition")
        elif app == "wallet" and function == "setup_wallet":
            wallets.append(command)
            groups.append("initialization")
        else:
            conflicts.append(f"command {index} is not allowed in an Instance base")

    if not 2 <= len(schemas) <= 3:
        conflicts.append("Instance must create 2 or 3 database tables")
    if not 2 <= len(files) <= 3:
        conflicts.append("Instance must create 2 or 3 TXT files")
    if len(profiles) != 1:
        conflicts.append("Instance must insert exactly one profile row")
    if len(wallets) != 1:
        conflicts.append("Instance must initialize Wallet exactly once")
    if "initialization" in groups and "definition" in groups[groups.index("initialization"):]:
        conflicts.append("definitions must precede initialization")

    filenames = [
        item.get("parameter", {}).get("name")
        for item in files
        if isinstance(item.get("parameter"), dict)
    ]
    if len(filenames) != len(files) or any(
        not isinstance(name, str) or not name.endswith(".txt") for name in filenames
    ):
        conflicts.append("every base File must have a valid .txt name")
    if len(filenames) != len(set(filenames)):
        conflicts.append("base File names must be unique")
    for command in schemas + files:
        if not isinstance(command.get("description"), str) or not command["description"].strip():
            conflicts.append("each table and File definition needs a description")
    return conflicts


class InstanceCreator:
    """Create one Instance with one LLM request."""

    def __init__(self, client=None, domains_dir=DOMAINS_DIR):
        self.client = client or LLMClient(temperature=0.7)
        self.domains_dir = Path(domains_dir)

    def create_instance(self, domain_spec, instance_id):
        domain_dir = self.domains_dir / domain_spec["domain_id"]
        instances_dir = domain_dir / "instances"
        instance_dir = instances_dir / instance_id
        if instance_dir.exists():
            raise FileExistsError(f"Instance already exists: {instance_dir}")

        existing = []
        if instances_dir.exists():
            for path in sorted(instances_dir.glob("*/description.json")):
                item = json.loads(path.read_text(encoding="utf-8"))
                existing.append(
                    {
                        "instance_id": path.parent.name,
                        "instance_description": item["instance_description"],
                        "personal_description": item["personal_description"],
                        "database_description": item["database_description"],
                        "file_description": item["file_description"],
                    }
                )
        used_protagonist_names = []
        for path in sorted(self.domains_dir.glob("*/instances/*/description.json")):
            personal = json.loads(path.read_text(encoding="utf-8"))[
                "personal_description"
            ]
            match = re.match(r"^(.+?)\s+(?:is|was)\b", personal)
            if match:
                used_protagonist_names.append(match.group(1))
        prompt = build_instance_creator_prompt(
            domain_spec,
            instance_id,
            existing,
            sorted(set(used_protagonist_names)),
            APPS_REFERENCE_PATH.read_text(encoding="utf-8"),
            load_plan()["design"]["benchmark_year"],
        )
        try:
            instance = json.loads(self.client.generate_json(prompt))
        except json.JSONDecodeError as error:
            raise ValueError(f"LLM returned invalid JSON: {error.msg}") from error
        conflicts = find_instance_conflicts(instance, instance_id)
        if conflicts:
            raise ValueError("Invalid Instance output: " + "; ".join(conflicts))

        instances_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{instance_id}.staging-", dir=instances_dir)
        )
        try:
            parser = CommandParser.for_instance(staging, instance_id)
            parser.execute_commands(instance)
            tables = parser.apps["database"].execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )["rows"]
            if not 2 <= len(tables) <= 3:
                raise ValueError("executed Instance must contain 2 or 3 tables")
            for (table,) in tables:
                columns = parser.apps["database"].execute(
                    f'PRAGMA table_info("{table}")'
                )["rows"]
                if len(columns) > 5:
                    raise ValueError(f"table {table!r} exceeds 5 columns")

            for trigger in TRIGGER_TYPES:
                (staging / "scenarios" / trigger / "logs").mkdir(parents=True)
            description = {
                "instance_description": instance["instance_description"],
                "personal_description": instance["personal_description"],
                "database_description": [
                    item["description"]
                    for item in instance["commands"]
                    if item["app"] == "database" and "description" in item
                ],
                "file_description": [
                    {
                        "name": item["parameter"]["name"],
                        "description": item["description"],
                    }
                    for item in instance["commands"]
                    if item["app"] == "files" and "description" in item
                ],
            }
            (staging / "description.json").write_text(
                json.dumps(description, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (staging / "instance_output.json").write_text(
                json.dumps(instance, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            staging.rename(instance_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return instance


def create_instance(domain_spec, instance_id):
    return InstanceCreator().create_instance(domain_spec, instance_id)
