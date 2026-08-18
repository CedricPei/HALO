"""Single LLM semantic review for the current Instance and candidate."""

import json

from llm_client import LLMClient
from .prompt import build_reviewer_prompt
from .scenario_creator import (
    DOMAINS_DIR,
    get_instance_dir,
    load_domain_scenario_context_summaries,
    load_instance_context,
)


class ReviewerOutputError(ValueError):
    """The Reviewer returned unusable JSON rather than semantic feedback."""


class Reviewer:
    """Review a candidate against the complete accepted Instance prefix."""

    def __init__(self):
        self.client = LLMClient(temperature=0)

    def review(
        self,
        domain_id,
        instance_id,
        expected_trigger_type,
        candidate_scenario,
    ):
        domain_dir = DOMAINS_DIR / domain_id
        instance_dir = get_instance_dir(domain_id, instance_id)
        domain_spec = json.loads(
            (domain_dir / "domain.json").read_text(encoding="utf-8")
        )
        description = json.loads(
            (instance_dir / "profile.json").read_text(encoding="utf-8")
        )
        accepted = [
            item
            for item in load_domain_scenario_context_summaries(domain_dir)
            if not (
                item["instance_id"] == instance_id
                and item["scenario_id"] == candidate_scenario["scenario_id"]
            )
        ]
        base_state = load_instance_context(instance_dir, instance_id)
        del base_state["description"]
        prompt = build_reviewer_prompt(
            domain_spec=domain_spec,
            instance_description=description,
            base_state=base_state,
            accepted_scenarios=accepted,
            candidate_scenario=candidate_scenario,
            expected_trigger_type=expected_trigger_type,
        )
        try:
            result = json.loads(
                self.client.generate_text(
                    prompt,
                    request_label=(
                        f"{domain_id}/{instance_id}/"
                        f"{candidate_scenario['scenario_id']}/Reviewer"
                    ),
                )
            )
        except json.JSONDecodeError as error:
            raise ReviewerOutputError(
                f"Reviewer returned invalid JSON: {error.msg}"
            ) from error
        if not isinstance(result, dict) or set(result) != {"valid", "issues"}:
            raise ReviewerOutputError("Reviewer must return exactly valid and issues")
        if not isinstance(result["valid"], bool):
            raise ReviewerOutputError("Reviewer valid must be a boolean")
        if not isinstance(result["issues"], list) or any(
            not isinstance(issue, str) or not issue.strip()
            for issue in result["issues"]
        ):
            raise ReviewerOutputError(
                "Reviewer issues must be a list of non-empty strings"
            )
        if result["valid"] is False and not result["issues"]:
            raise ReviewerOutputError(
                "Reviewer must provide at least one issue when valid is false"
            )
        return {
            "valid": result["valid"] is True and not result["issues"],
            "issues": result["issues"],
        }
