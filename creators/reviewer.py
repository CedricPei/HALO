"""Single LLM semantic review for the current Instance and candidate."""

import json
from pathlib import Path

from .llm_client import LLMClient
from .prompt import build_reviewer_prompt
from .scenario_creator import (
    DOMAINS_DIR,
    get_instance_dir,
    load_accepted_scenarios,
    load_instance_context,
)


def _base_state(instance_dir, instance_id):
    return load_instance_context(instance_dir, instance_id)


class Reviewer:
    """Review a candidate against the complete accepted Instance prefix."""

    def __init__(self, client=None, domains_dir=DOMAINS_DIR):
        self.client = client or LLMClient(temperature=0)
        self.domains_dir = Path(domains_dir)

    def review(
        self,
        domain_id,
        instance_id,
        expected_trigger_type,
        candidate_scenario,
    ):
        domain_dir = self.domains_dir / domain_id
        instance_dir = get_instance_dir(self.domains_dir, domain_id, instance_id)
        domain_spec = json.loads(
            (domain_dir / "domain.json").read_text(encoding="utf-8")
        )
        description = json.loads(
            (instance_dir / "description.json").read_text(encoding="utf-8")
        )
        prompt = build_reviewer_prompt(
            domain_spec=domain_spec,
            instance_description=description,
            base_state=_base_state(instance_dir, instance_id),
            accepted_scenarios=load_accepted_scenarios(instance_dir),
            candidate_scenario=candidate_scenario,
            expected_trigger_type=expected_trigger_type,
        )
        try:
            result = json.loads(self.client.generate_json(prompt))
        except json.JSONDecodeError as error:
            raise ValueError(f"Reviewer returned invalid JSON: {error.msg}") from error
        if list(result) != ["valid", "issues"] or not isinstance(
            result.get("issues"), list
        ):
            raise ValueError("Reviewer must return exactly valid and issues")
        return {
            "valid": result.get("valid") is True and not result["issues"],
            "issues": result["issues"],
        }
