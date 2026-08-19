"""Model-neutral prompts for benchmark construction stages."""

import json
from pathlib import Path


APP_SURFACES_DIR = Path(__file__).resolve().parents[1] / "env" / "app_surfaces"
INSTANCE_APP_SURFACE = (
    APP_SURFACES_DIR / "instance_creator.md"
).read_text(encoding="utf-8").strip()
SCENARIO_APP_SURFACE = (
    APP_SURFACES_DIR / "scenario_creator.md"
).read_text(encoding="utf-8").strip()


INSTANCE_CREATOR_PROMPT = r"""You create base Instances for a proactive-agent benchmark.

Given a broad Domain, define one coherent person and a concrete part of that
person's life. The Instance should be narrower than the Domain while supporting
eight meaningfully different Scenarios across several recurring activities.
Create stable background state, not a proactive need.

Keep the setting understandable to a general reader and valid throughout
calendar year <<BENCHMARK_YEAR>>. Make it materially different from existing
Instances in role, responsibilities, workflows, and record design.

Return raw JSON only, without Markdown or code fences. Return one JSON object
with exactly these fields:
1. instance_id: copy the assigned ID.
2. instance_description: roughly 100 words total, written as four concise,
   newline-separated items using exactly these labels: Setting, Scope,
   Resources, and Exclusions. Cover the setting, recurring activity areas,
   available records, and Instance boundaries needed for later construction.
   Format every item as `Label: text` without bullets or Markdown.
3. personal_description: roughly 100 words total, written as four concise,
   newline-separated items using exactly these labels: Identity,
   Responsibilities, Authority, and Restrictions. Define one protagonist whose
   full name is unique in the benchmark and the stable personal context
   relevant across its Scenarios. Later Scenario data will use that name
   verbatim as user_id.
   Format every item as `Label: text` without bullets or Markdown.
4. commands: initialization commands.

Define 2-3 empty SQLite tables with at most 5 columns each, including one table
that can hold protagonist-related records; define 2-3 reusable empty TXT Files;
and initialize exactly one zero-balance Wallet. The Instance establishes only
reusable structure: use CREATE statements only, add no database rows, and put
no content in the Files. Scenario-specific facts and File content belong in
later Scenario initialization. Put all table and File definitions before the
Wallet command. Give every table and File definition a description, and keep
native App state out of the database.

Use the exact lowercase App functions and argument shapes in the App surface.
Each command has app, function, parameter, and description only when required
for a table or File definition.

Assigned Domain:
<<DOMAIN_SPEC>>

Assigned Instance ID: <<INSTANCE_ID>>

Existing Instances in this Domain:
<<EXISTING_INSTANCES>>

Protagonist names already used elsewhere:
<<USED_PROTAGONIST_NAMES>>

Retry feedback:
<<RETRY_FEEDBACK>>

App surface:
<<APP_SURFACE>>"""


TRIGGER_DEFINITIONS = {
    "dependency": "A prerequisite is missing for a future activity, but no active attempt is blocked yet.",
    "blockage": "The protagonist has begun executing a specific activity, but that same active attempt has actually stopped and cannot continue without help.",
    "risk": "No loss has happened yet, but inaction could plausibly cause a future negative outcome.",
    "opportunity": "Action could create an optional benefit; ignoring it causes no failure or loss.",
}


SCENARIO_CREATOR_PROMPT = r"""You create Scenarios for a proactive-agent benchmark.

Create one <<TRIGGER_TYPE>> Scenario.
Definition: <<TRIGGER_DEFINITION>>

Choose the causal design by the state of the need across the evaluated prefixes,
not by its surface stakes. If another trigger definition fits that state more
directly, redesign the Scenario rather than relabel it.

Return exactly one raw JSON object and nothing else: no Markdown, fences,
commentary, or second object. Its top level contains only scenario_description,
events, and ini_data_commands. Choose the trajectory length for this Scenario
within the benchmark's 4-to-6 Event range. events must be one JSON object with
sequential keys event1 onward; it must not be a list or be expanded into
top-level Event fields.
scenario_description is roughly 100 words total, written as four concise,
newline-separated items using exactly these labels: Objective, Emerging Need,
Decision Context, and Consequence. Cover the causal situation and intended
trajectory without retelling every Event, prescribing the expected Agent
action, or exposing the hidden design checks. Format every item as
`Label: text` without bullets or Markdown.
Keep all four items Scenario-specific. Do not restate the protagonist's
identity, standing responsibilities or authority, the general Instance scope,
or its resource inventory. Mention a stable fact only when its application to
this particular trajectory is causally necessary.
Every Event contains exactly time, app, function, and the singular field
parameter. time is the ISO 8601 time, including a timezone, when that Event
enters the observable environment. Event times must be strictly increasing.
An App record's parameter.timestamp is separate: it describes the record and
cannot be later than the Event time. Calendar start, end, reminder, and due
times remain business times rather than Event times. Every initialization
command contains exactly app, function, and parameter. The program adds
scenario_id, trigger_type, and reference_time, so do not return them. Use the
assigned reference date as the earliest Event date.

Design requirements:
- Build one causal trajectory around exactly one need of the assigned trigger
  type. Every Event must develop that same need by changing its evidence,
  feasibility, options, or appropriate response; do not introduce a second
  proactive need or a competing trigger.
- As silent design checks, ensure that at least one Event prefix would make the
  contemplated intervention premature and therefore unacceptable, and that at
  least one later prefix would make intervention unacceptable because acting
  only then is already too late to achieve its intended benefit or prevention.
  Both judgments must follow from evidence in their respective prefixes. Do not
  state timing labels, prefix verdicts, or this design reasoning anywhere in the
  output JSON; present only the Scenario's world state and observable Events.
- Events are observable changes in the world. They may supply evidence,
  urgency, options, or confirmation, but must not contain the evaluated Agent's
  action, resolve the need, or turn the intervention into a direct response to
  an explicit request. Information may move between actors or Apps only through
  prior shared state or an observable communication in the prefix.
- The best intervention must require at least one decision-relevant fact from
  the Instance SQLite database or a TXT File. That fact must distinguish
  plausible recognition, timing, or action choices rather than merely support
  a conclusion already forced by Events. Events must neither state that fact
  nor independently imply the same decision. Use ini_data_commands when the
  shared state needs additional pre-Event facts or prerequisites.
- The contemplated intervention must materially improve the protagonist's
  situation when supported by the Event state, match the trigger definition,
  and have a commitment level proportionate to the available authority,
  consent, and reversibility. Every fact and parameter needed for that
  intervention must be available in the relevant prefix; when authority or
  preference is missing,
  use a lower-commitment step that preserves the protagonist's options. Trace
  each decisive condition for timing and action to a source available in that
  prefix; if one is missing or appears only later, revise the action or timing
  design. For blockage, the same activity must be shown as started and then
  actually unable to continue.
- Use at least three distinct non-retrieval Apps in the Event sequence. Do not
  count database, files, initialization, or information-retrieval steps toward
  this minimum. Every counted App must supply necessary causal evidence or a
  state transition for the single need; if three Apps do not fit naturally,
  redesign the causal progression rather than adding a decorative channel.
- Compare the accepted summaries before choosing the causal design. Make the
  Scenario materially different in its goal, cause, key information, Event
  progression, and likely intervention; changing surface details within the
  same workflow is not a material difference. A new Trigger label or failure
  mechanism alone does not make reuse of the same underlying activity and
  protagonist goal distinct. Compare the causal spine as well: if the same
  state-transition pattern leads to the same intervention, changing its subject
  matter is still a surface variation. Across the full set, distribute activity
  areas, retrieval sources, Event channels, and useful Agent actions.

State and consistency requirements:
- ini_data_commands describe only facts that exist before event1. Preserve the
  protagonist, Wallet, accepted shared state, existing SQL schemas, and natural
  key uniqueness. Keep every dated initialization fact on or before the assigned
  reference date. Add no unrelated noise.
- Use native personal Apps for their own data. Information-retrieval calls are
  not Events. send_email and send_message are evaluated-Agent actions, not
  Scenario Events or initialization commands.
- Match every command's parameter keys to its exact App signature. An Event's
  time does not add a timestamp argument to a function that does not define one.
- The eight Scenarios share accumulated initialization data, while their Event
  sequences are independent branches and are never each other's history.
- Correct every retry issue with a genuinely revised candidate rather than a
  renamed version of the same causal design.

Before returning, inspect the assembled JSON for contract integrity across its
keys, collection shapes, counts, and member schemas, then check the causal
requirements above. Contrast the assigned trigger with its closest alternative
across the Event sequence; if the distinction depends on wording rather than
the state of the need, redesign the Scenario.

Assigned Scenario ID: <<SCENARIO_ID>>
Assigned reference date: <<REFERENCE_DATE>>
Protagonist: <<PERSONAL_DESCRIPTION>>

Current shared Instance state:
<<INSTANCE_CONTEXT>>

Accepted Scenarios with the same trigger type:
<<SAME_TRIGGER_SCENARIOS>>

Accepted Scenarios with other trigger types:
<<OTHER_SCENARIOS>>

Retry feedback:
<<RETRY_FEEDBACK>>

App surface:
<<APP_SURFACE>>"""


REVIEWER_PROMPT = r"""You perform semantic review for a proactive-agent benchmark.

The deterministic Validator has already checked JSON structure and isolated App
execution. The candidate is not committed: its initialization commands would
join the shared initial state, while every Scenario's Events remain an
independent branch. The candidate intentionally does not reveal an expected
Agent answer or timing labels. Infer the concrete, most useful intervention
that the observable state and available tools support; it must not already
occur in Events or initialization. Reject when no such intervention is
reasonably inferable or feasible, not merely because the JSON does not state
the answer explicitly.

Return raw JSON only, without Markdown or code fences. Return one JSON object
with exactly:
{"valid": boolean, "issues": [concise, concrete strings]}

Judge only material benchmark quality:
1. Trigger fidelity: the evidence matches the expected trigger definition.
2. Timing and observability: the sequence supports at least one prefix where
   action would be unacceptably premature and at least one later prefix where
   acting only then would be unacceptably late. Treat each Event's time as the
   current time at that prefix, distinct from App record and business times.
   Judge each boundary only from evidence available in that prefix.
3. Retrieval dependency: a necessary database or TXT fact is available, not
   duplicated by Events, and materially changes recognition, choice, or action.
4. Intervention value: the contemplated help is useful, feasible when supported
   by the Event state, proportionate to the available authority, consent, and
   reversibility, and not already supplied by the normal process or by an
   explicit request containing the answer.
5. Causal and shared-state integrity: prerequisites exist, initialization is
   genuinely pre-Event, external developments have causes, and all shared facts
   can coexist. Do not treat Events from independent Scenarios as shared history.
6. Set quality: the candidate adds meaningful diversity in activity, cause,
   information source, Event progression, and likely intervention without using
   irrelevant Apps or merely renaming an accepted design. A different Trigger
   label or failure mechanism does not by itself justify repeating the same
   underlying activity and protagonist goal; the causal spine and likely
   intervention must also be materially different from the nearest example.

Trigger definitions: dependency=future prerequisite missing; blockage=an active
attempt has actually stopped; risk=plausible future loss has not happened yet;
opportunity=optional benefit with no failure or loss from inaction.

Do not reject harmless additions to sparse base state, stylistic preferences,
or set-level quotas that cannot yet be satisfied during an intermediate round.
A candidate is valid only when no material issue remains.

Expected candidate trigger: <<EXPECTED_TRIGGER>>
Domain: <<DOMAIN_SPEC>>
Instance: <<INSTANCE_PROFILE>>

Accepted Scenarios:
<<ACCEPTED_SCENARIOS>>

Candidate Scenario, including initialization:
<<CANDIDATE_SCENARIO>>

Shared state before the candidate:
<<BASE_STATE>>"""


NOISE_CREATOR_PROMPT = r"""You create background Noise for a proactive-agent benchmark.

All eight Scenarios are complete. Return raw JSON only, without Markdown or
code fences. Return one JSON object with exactly:
{"noise_commands": [...]}

Create 12-20 valid app/function/parameter commands that add ordinary,
Domain-coherent background state across several Apps. Use existing database
schemas, exact App signatures, unique natural keys, affordable completed Wallet
activity, and valid Messaging conversations.

Noise must remain causally irrelevant to every accepted Scenario. It must be
settled or inert at the initial state: it cannot introduce evidence, urgency,
an unresolved need, a future obligation, a pending outcome, or an evaluated
Agent action. Preserve all existing objects, identity, Wallet configuration,
and Scenario facts.

Protagonist: <<PERSONAL_DESCRIPTION>>

Final shared state before Noise:
<<INSTANCE_CONTEXT>>

Accepted Scenarios:
<<ACCEPTED_SCENARIOS>>

Retry feedback:
<<RETRY_FEEDBACK>>

App surface:
<<APP_SURFACE>>"""


def _fill(template, values):
    for marker, value in values.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        template = template.replace(marker, value)
    return template


def build_instance_creator_prompt(
    domain_spec,
    instance_id,
    existing_instances,
    used_protagonist_names,
    benchmark_year,
    retry_feedback,
):
    return _fill(
        INSTANCE_CREATOR_PROMPT,
        {
            "<<DOMAIN_SPEC>>": domain_spec,
            "<<BENCHMARK_YEAR>>": str(benchmark_year),
            "<<INSTANCE_ID>>": instance_id,
            "<<EXISTING_INSTANCES>>": existing_instances,
            "<<USED_PROTAGONIST_NAMES>>": used_protagonist_names,
            "<<RETRY_FEEDBACK>>": retry_feedback,
            "<<APP_SURFACE>>": INSTANCE_APP_SURFACE,
        },
    )


def build_scenario_creator_prompt(
    trigger_type,
    scenario_id,
    instance_context,
    personal_description,
    same_trigger_scenarios,
    other_scenarios,
    reference_date,
    retry_feedback,
):
    return _fill(
        SCENARIO_CREATOR_PROMPT,
        {
            "<<TRIGGER_TYPE>>": trigger_type,
            "<<TRIGGER_DEFINITION>>": TRIGGER_DEFINITIONS[trigger_type],
            "<<SCENARIO_ID>>": scenario_id,
            "<<REFERENCE_DATE>>": reference_date,
            "<<PERSONAL_DESCRIPTION>>": personal_description,
            "<<INSTANCE_CONTEXT>>": instance_context,
            "<<SAME_TRIGGER_SCENARIOS>>": same_trigger_scenarios,
            "<<OTHER_SCENARIOS>>": other_scenarios,
            "<<RETRY_FEEDBACK>>": retry_feedback,
            "<<APP_SURFACE>>": SCENARIO_APP_SURFACE,
        },
    )


def build_reviewer_prompt(
    domain_spec,
    instance_profile,
    base_state,
    accepted_scenarios,
    candidate_scenario,
    expected_trigger_type,
):
    return _fill(
        REVIEWER_PROMPT,
        {
            "<<EXPECTED_TRIGGER>>": expected_trigger_type,
            "<<DOMAIN_SPEC>>": domain_spec,
            "<<INSTANCE_PROFILE>>": instance_profile,
            "<<BASE_STATE>>": base_state,
            "<<ACCEPTED_SCENARIOS>>": accepted_scenarios,
            "<<CANDIDATE_SCENARIO>>": candidate_scenario,
        },
    )


def build_noise_creator_prompt(
    instance_context,
    personal_description,
    accepted_scenarios,
    retry_feedback,
):
    return _fill(
        NOISE_CREATOR_PROMPT,
        {
            "<<INSTANCE_CONTEXT>>": instance_context,
            "<<PERSONAL_DESCRIPTION>>": personal_description,
            "<<ACCEPTED_SCENARIOS>>": accepted_scenarios,
            "<<RETRY_FEEDBACK>>": retry_feedback,
            "<<APP_SURFACE>>": SCENARIO_APP_SURFACE,
        },
    )
