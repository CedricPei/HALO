"""Generate and initialize one concrete Instance inside a human Domain."""

import json
import re
import shutil
import tempfile
from pathlib import Path

from command_parser import CommandParser
from llm_client import LLMClient

from .prompt import TRIGGER_DEFINITIONS, build_instance_creator_prompt


ROOT_DIR = Path(__file__).resolve().parents[1]
DOMAINS_DIR = ROOT_DIR / "domains"


def _protagonist_name(personal_description):
    personal = personal_description.strip()
    if personal.startswith("Identity:"):
        personal = personal.removeprefix("Identity:").strip()
    match = re.match(r"^(.+?)(?=,\s*age\b|\s+(?:is|was)\b)", personal)
    return match.group(1).strip() if match else None


def _database_sql(parameter):
    if isinstance(parameter, str):
        return parameter
    if (
        isinstance(parameter, dict)
        and set(parameter).issubset({"sql", "parameters"})
        and isinstance(parameter.get("sql"), str)
    ):
        return parameter["sql"]
    return None


def find_instance_conflicts(instance, expected_instance_id):
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
    if set(instance) != set(fields):
        conflicts.append(f"top-level fields must be exactly {fields}")

    instance_id = instance.get("instance_id")
    if not isinstance(instance_id, str) or not re.fullmatch(
        r"[a-z][a-z0-9_]*", instance_id
    ):
        conflicts.append("instance_id must be lowercase snake_case")
    if instance_id != expected_instance_id:
        conflicts.append(f"instance_id must equal {expected_instance_id!r}")
    for field in ("instance_description", "personal_description"):
        if not isinstance(instance.get(field), str) or not instance[field].strip():
            conflicts.append(f"{field} must be a non-empty string")

    commands = instance.get("commands")
    if not isinstance(commands, list):
        return conflicts + ["commands must be a list"]

    schemas, files, wallets, groups = [], [], [], []
    for index, command in enumerate(commands, 1):
        if not isinstance(command, dict):
            conflicts.append(f"command {index} must be an object")
            continue
        app, function, parameter = (
            command.get("app"),
            command.get("function"),
            command.get("parameter"),
        )
        if app == "database" and function == "execute":
            sql = _database_sql(parameter)
            if sql is None:
                conflicts.append(
                    f"command {index} database.execute needs SQL as a string or "
                    "an object with sql and optional parameters"
                )
                continue
            verb = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else ""
            if verb == "CREATE":
                schemas.append(command)
                groups.append("definition")
            else:
                conflicts.append(f"command {index} must be a CREATE table definition")
        elif app == "files" and function == "create_file":
            files.append(command)
            groups.append("definition")
        elif app == "wallet" and function == "setup_wallet":
            wallets.append(command)
            groups.append("initialization")
        else:
            conflicts.append(
                f"command {index} is not allowed in an Instance base: got "
                f"{app!r}.{function!r}, fields {sorted(command)}, and "
                f"parameter type {type(parameter).__name__}"
            )

    if not 2 <= len(schemas) <= 3:
        conflicts.append("Instance must create 2 or 3 database tables")
    if not 2 <= len(files) <= 3:
        conflicts.append("Instance must create 2 or 3 TXT files")
    if len(wallets) != 1:
        conflicts.append("Instance must initialize Wallet exactly once")
    if (
        "initialization" in groups
        and "definition" in groups[groups.index("initialization") :]
    ):
        conflicts.append("definitions must precede initialization")

    file_parameters = [item.get("parameter") for item in files]
    filenames = [
        parameter.get("name")
        for parameter in file_parameters
        if isinstance(parameter, dict)
    ]
    if len(filenames) != len(files) or any(
        not isinstance(name, str) or not name.endswith(".txt") for name in filenames
    ):
        conflicts.append("every base File must have a valid .txt name")
    if len(filenames) != len(set(filenames)):
        conflicts.append("base File names must be unique")
    if any(
        not isinstance(parameter, dict)
        or set(parameter) not in ({"name"}, {"name", "content"})
        or parameter.get("content", "") != ""
        for parameter in file_parameters
    ):
        conflicts.append("base Files must be definitions with no initial content")
    for command in wallets:
        parameter = command.get("parameter")
        if not isinstance(parameter, dict) or parameter.get("balance", 0) != 0:
            conflicts.append("the base Wallet must have a zero balance")
    for command in schemas + files:
        if (
            not isinstance(command.get("description"), str)
            or not command["description"].strip()
        ):
            conflicts.append("each table and File definition needs a description")
    return conflicts


class InstanceCreator:
    """Create one Instance with one LLM request."""

    def __init__(self):
        self.client = LLMClient(temperature=0.7)

    def create_instance(
        self,
        domain_spec,
        instance_id,
        benchmark_year,
        retry_feedback,
    ):
        domain_dir = DOMAINS_DIR / domain_spec["domain_id"]
        instances_dir = domain_dir / "instances"
        instance_dir = instances_dir / instance_id
        if instance_dir.exists():
            raise FileExistsError(f"Instance already exists: {instance_dir}")

        existing, names_in_existing = [], set()
        for path in sorted(instances_dir.glob("*/profile.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            name = _protagonist_name(item["personal_description"])
            if name:
                names_in_existing.add(name)
            output = json.loads(
                (path.parent / "outputs" / "instance.json").read_text(
                    encoding="utf-8"
                )
            )
            existing.append(
                {
                    "instance_id": path.parent.name,
                    "instance_description": item["instance_description"],
                    "personal_description": item["personal_description"],
                    "database_description": [
                        command["description"]
                        for command in output["commands"]
                        if command["app"] == "database"
                    ],
                    "file_description": [
                        {
                            "name": command["parameter"]["name"],
                            "description": command["description"],
                        }
                        for command in output["commands"]
                        if command["app"] == "files"
                    ],
                }
            )
        used_protagonist_names = []
        for path in sorted(DOMAINS_DIR.glob("*/instances/*/profile.json")):
            personal = json.loads(path.read_text(encoding="utf-8"))[
                "personal_description"
            ]
            name = _protagonist_name(personal)
            if name and name not in names_in_existing:
                used_protagonist_names.append(name)
        prompt = build_instance_creator_prompt(
            domain_spec,
            instance_id,
            existing,
            sorted(set(used_protagonist_names)),
            benchmark_year,
            retry_feedback=retry_feedback,
        )
        try:
            instance = json.loads(
                self.client.generate_text(
                    prompt,
                    request_label=(
                        f"{domain_spec['domain_id']}/{instance_id}/Instance"
                    ),
                )
            )
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

            if not _protagonist_name(instance["personal_description"]):
                raise ValueError(
                    "personal_description must begin with the protagonist's full name"
                )

            (staging / "outputs").mkdir()
            (staging / "scenarios" / "logs").mkdir(parents=True)
            for trigger in TRIGGER_DEFINITIONS:
                (staging / "scenarios" / trigger).mkdir()
            description = {
                "instance_description": instance["instance_description"],
                "personal_description": instance["personal_description"],
            }
            (staging / "profile.json").write_text(
                json.dumps(description, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (staging / "outputs" / "instance.json").write_text(
                json.dumps(instance, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            staging.rename(instance_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return instance
