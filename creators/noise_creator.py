"""Add one final batch of unrelated background data to an Instance."""

import json
from pathlib import Path

from command_parser import CommandParser

from .llm_client import LLMClient
from .prompt import build_noise_creator_prompt
from .scenario_creator import (
    APPS_REFERENCE_PATH,
    DOMAINS_DIR,
    get_instance_dir,
    load_accepted_scenarios,
    load_instance_context,
    load_personal_description,
)
from .validator import validate_noise


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


class NoiseCreator:
    """Generate and commit final Noise with one LLM request."""

    def __init__(self, client=None, domains_dir=DOMAINS_DIR):
        self.client = client or LLMClient(temperature=0.7)
        self.domains_dir = Path(domains_dir)

    def create_noise(self, domain_id, instance_id):
        instance_dir = get_instance_dir(self.domains_dir, domain_id, instance_id)
        prompt = build_noise_creator_prompt(
            instance_context=load_instance_context(instance_dir, instance_id),
            personal_description=load_personal_description(instance_dir),
            accepted_scenarios=load_accepted_scenarios(instance_dir),
            app_reference=APPS_REFERENCE_PATH.read_text(encoding="utf-8"),
        )
        try:
            noise = json.loads(self.client.generate_json(prompt))
        except json.JSONDecodeError as error:
            raise ValueError(f"Noise Creator returned invalid JSON: {error.msg}") from error

        _write_json(instance_dir / "noise_output.json", noise)
        return self._validate_and_commit(instance_dir, instance_id, noise)

    def commit_saved_noise(self, domain_id, instance_id):
        """Revalidate a saved LLM output without making another LLM request."""
        instance_dir = get_instance_dir(self.domains_dir, domain_id, instance_id)
        noise = json.loads(
            (instance_dir / "noise_output.json").read_text(encoding="utf-8")
        )
        return self._validate_and_commit(instance_dir, instance_id, noise)

    def _validate_and_commit(self, instance_dir, instance_id, noise):
        conflicts = validate_noise(instance_dir, instance_id, noise)
        validation = {"valid": not conflicts, "conflicts": conflicts}
        if conflicts:
            _write_json(instance_dir / "noise_validation.json", validation)
            raise ValueError("Invalid Noise output: " + "; ".join(conflicts))

        CommandParser.for_instance(instance_dir, instance_id).execute_commands(
            {"commands": noise["noise_commands"]}
        )
        _write_json(instance_dir / "noise_validation.json", validation)
        return {"noise": noise, "validation": validation}


def create_noise(domain_id, instance_id):
    return NoiseCreator().create_noise(domain_id, instance_id)
