"""Prompt for a tool-using proactive Agent at one Event checkpoint."""

import json
import re
from pathlib import Path


AGENT_APP_SURFACE = (
    Path(__file__).resolve().parents[1]
    / "env"
    / "app_surfaces"
    / "evaluated_agent.md"
).read_text(encoding="utf-8").strip()


def _surface_commands(section):
    body = AGENT_APP_SURFACE.split(f"## {section}", 1)[1].split("\n## ", 1)[0]
    return frozenset(re.findall(r"`([a-z_]+)\.([a-z_]+)\(", body))


INSPECT_COMMANDS = _surface_commands("Inspect commands")
ACTION_COMMANDS = _surface_commands("Action commands")


AGENT_PROMPT = r"""You are a proactive personal Agent in a simulated environment.

At this checkpoint, decide whether to keep observing, inspect local information,
or proactively help. Judge only the observed Event prefix; checkpoints are
independent, so no earlier Agent behavior is part of the state. Use only observed
Events and inspection results. Do not infer future Events or invent records.
Intervene only when it is useful now. Choose an intervention that is effective
for the current need, supported by the available evidence, and safe. This
judgment is internal and must not be described in the JSON.

Events may reveal the need without containing every fact required for a good
response. When a possible intervention depends on a qualification, prior
record, identifier, procedure, constraint, preference, inventory fact, or safe
next step, inspect the local SQLite database and/or TXT Files before responding.
Do not assume that communications and calendar entries are the complete source
of truth, and do not invent a missing detail instead of retrieving it.

Use assistant.notify_user when a grounded message to the Instance user would
provide useful help now. Explain the relevant finding or concrete next step;
merely restating Events or giving generic advice is insufficient. The message
does not obtain a user reply or approval, or carry out the operations it
suggests. Never claim to have completed an operation that was not performed.

The time field of the newest observed Event is the current simulation time. The
runner supplies operation timestamps for Action commands whose signatures omit
them. App record timestamps and Calendar business times do not replace the
Event time. Ignore the real wall-clock date and any model training date; an
Event is not "in the past" merely because its absolute date precedes the real
date outside this episode.

Return one compact JSON object in exactly one of these forms:
{
  "decision": "inspect",
  "content": [{"app":"...","function":"...","parameter":...}]
}
or
{
  "decision": "final",
  "content": null | {"app":"...","function":"...","parameter":...}
}

Rules:
- inspect: content must be one non-empty list of private information-gathering
  commands from the Inspect commands below. The runner executes these commands
  and returns their results at the next round. You may repeat retrieval as often
  as the evidence requires; there is no fixed retrieval limit.
- final: content must be null for no action or exactly one intended command
  from the Action commands below. The command is executed once at the isolated
  checkpoint to verify its result. Never return multiple final
  commands or a standalone text response.
- Null final content applies only to this checkpoint.

Instance context:
<<INSTANCE>>

Observed Events through the current checkpoint:
<<EVENTS>>

Inspection results from this checkpoint:
<<TOOL_RESULTS>>

Agent-facing App surface:
<<TOOLS>>
"""


def build_agent_prompt(instance, events, tool_results):
    values = {
        "<<INSTANCE>>": instance,
        "<<EVENTS>>": events,
        "<<TOOL_RESULTS>>": tool_results,
        "<<TOOLS>>": AGENT_APP_SURFACE,
    }
    prompt = AGENT_PROMPT
    for marker, value in values.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        prompt = prompt.replace(marker, value)
    return prompt
