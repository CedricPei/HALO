"""Convert JSON commands into calls on initialized App objects."""

import json
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
        return cls(
            {
                "database": Database(data_dir / f"{instance_id}.sqlite"),
                "files": Files(data_dir),
                "email": Email(app_dir / "email.json"),
                "calendar": Calendar(app_dir / "calendar.json"),
                "messaging": Messaging(app_dir / "messaging.json"),
                "contacts": Contacts(app_dir / "contacts.json"),
                "wallet": Wallet(app_dir / "wallet.json"),
            }
        )

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

    def execute_commands(self, data: dict | str) -> list:
        """Read commands from a JSON object and execute them in order."""
        if isinstance(data, str):
            data = json.loads(data)
        commands = data["commands"]
        return [self.execute_command(command) for command in commands]
