"""Convert JSON commands into calls on initialized App objects."""

import json
import re
from pathlib import Path

from env.apps import Calendar, Contacts, Database, Email, Files, Messaging, Wallet


class CommandParser:
    def __init__(self, apps: dict[str, object]):
        self.apps = apps

    @classmethod
    def for_instance(cls, instance_dir, instance_id):
        """Create all Apps for one Instance environment."""
        instance_dir = Path(instance_dir)
        data_dir = instance_dir / "data"
        app_dir = instance_dir / "app"
        owner = cls._instance_owner(instance_dir)
        return cls(
            {
                "database": Database(data_dir / f"{instance_id}.sqlite"),
                "files": Files(data_dir),
                "email": Email(app_dir / "email.json", owner=owner),
                "calendar": Calendar(app_dir / "calendar.json"),
                "messaging": Messaging(
                    app_dir / "messaging.json",
                    owner=cls._messaging_owner(app_dir / "messaging.json", owner),
                ),
                "contacts": Contacts(app_dir / "contacts.json"),
                "wallet": Wallet(app_dir / "wallet.json"),
            }
        )

    @staticmethod
    def _instance_owner(instance_dir):
        """Read the protagonist name from the Instance personal description."""
        description_path = Path(instance_dir) / "profile.json"
        if not description_path.exists():
            return None
        description = json.loads(description_path.read_text(encoding="utf-8"))
        personal = description.get("personal_description", "").strip()
        if personal.startswith("Identity:"):
            personal = personal.removeprefix("Identity:").strip()
        match = re.match(r"^(.+?)(?=,\s*age\b|\s+(?:is|was)\b)", personal)
        if not match:
            raise ValueError(
                "personal_description must begin with the protagonist's full name"
            )
        return match.group(1).strip()

    @staticmethod
    def _messaging_owner(state_path, owner):
        """Resolve a stored account's short name or email without changing history."""
        if not owner or not state_path.exists():
            return owner
        conversations = json.loads(state_path.read_text(encoding="utf-8")).get(
            "conversations", []
        )
        if not conversations:
            return owner
        common = set.intersection(
            *(set(item["participants"]) for item in conversations)
        )
        if owner in common:
            return owner
        # A unique shared participant must also match the protagonist's name.
        # Never select an unrelated contact merely because it appears often.
        if len(common) == 1:
            candidate = next(iter(common))
            names = owner.casefold().split()
            account = candidate.casefold().split("@", 1)[0]
            tokens = re.split(r"[\s._-]+", account)
            if len(names) >= 2 and tokens == [names[0], names[-1]]:
                return candidate
        return owner

    def execute_command(self, command: dict):
        """Execute one command and return the App function result."""
        app = self.apps[command["app"]]
        function = getattr(app, command["function"])
        parameter = command.get("parameter")

        if isinstance(parameter, dict):
            result = function(**parameter)
        elif isinstance(parameter, list):
            result = function(*parameter)
        elif parameter is None:
            result = function()
        else:
            result = function(parameter)

        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(
                f"{command['app']}.{command['function']} failed: {result['error']}"
            )
        return result

    def execute_commands(self, data: dict) -> list:
        """Execute the commands in one parsed JSON object."""
        return [self.execute_command(command) for command in data["commands"]]
