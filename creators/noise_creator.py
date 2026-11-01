"""Add one final batch of unrelated background data to an Instance."""

import json
import logging
from concurrent.futures import CancelledError

from command_parser import CommandParser
from llm_client import LLMClient, LLMTransportError

from .prompt import build_noise_creator_prompt
from .scenario_creator import (
    get_instance_dir,
    load_accepted_scenarios,
    load_instance_profile,
    load_instance_state,
)
from .validator import validate_noise


LOGGER = logging.getLogger(__name__)


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


class NoiseCreator:
    """Generate revised Noise candidates and commit the first valid one."""

    def __init__(self):
        self.client = LLMClient(temperature=0.7)

    def create_noise(self, domain_id, instance_id, stop_event):
        instance_dir = get_instance_dir(domain_id, instance_id)
        profile = load_instance_profile(instance_dir)
        instance_context = {
            "overview": profile["instance_description"],
            **load_instance_state(instance_dir, instance_id),
        }
        accepted_scenarios = [
            {
                "scenario_id": scenario["scenario_id"],
                "trigger_type": scenario["trigger_type"],
                "scenario_description": scenario["scenario_description"],
                "events": scenario["events"],
            }
            for scenario in load_accepted_scenarios(instance_dir)
        ]
        feedback = []
        for attempt in range(1, 6):
            if stop_event.is_set():
                raise CancelledError
            LOGGER.info(
                "[%s/%s/Noise] semantic attempt %s/%s started",
                domain_id,
                instance_id,
                attempt,
                5,
            )
            prompt = build_noise_creator_prompt(
                instance_context=instance_context,
                personal_description=profile["personal_description"],
                accepted_scenarios=accepted_scenarios,
                retry_feedback=feedback,
            )
            try:
                raw_output = self.client.generate_text(
                    prompt,
                    request_label=f"{domain_id}/{instance_id}/Noise",
                )
                if stop_event.is_set():
                    raise CancelledError
                noise = json.loads(raw_output)
                conflicts = validate_noise(instance_dir, instance_id, noise)
                if conflicts:
                    raise ValueError("Invalid Noise output: " + "; ".join(conflicts))
                CommandParser.for_instance(
                    instance_dir, instance_id
                ).execute_commands({"commands": noise["noise_commands"]})
                _write_json(instance_dir / "outputs" / "noise.json", noise)
            except json.JSONDecodeError as error:
                failure = f"Noise Creator returned invalid JSON: {error.msg}"
            except ValueError as error:
                failure = str(error)
            except LLMTransportError:
                stop_event.set()
                raise
            else:
                return noise

            feedback.append(failure)
            LOGGER.warning(
                "[%s/%s/Noise] semantic attempt %s/%s failed retry=%s " "error=%s",
                domain_id,
                instance_id,
                attempt,
                5,
                attempt < 5,
                failure,
            )
        raise ValueError(
            f"Noise creation failed after 5 candidates: {feedback[-1]}"
        )
