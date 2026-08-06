"""Concise prompts for the four LLM stages."""

import json


INSTANCE_CREATOR_PROMPT = r"""You are the Instance Creator for a proactive-agent benchmark.

A human fixed a broad Domain. Create one concrete person and a coherent part of
that person's life inside it. The Instance must be narrower than the Domain but
broad enough for eight different Scenarios. Do not create a proactive need.
The benchmark is situated in <<BENCHMARK_YEAR>>. Keep cohort years, roles, and
stable temporal context consistent with that year.

Choose an ordinary, broadly understandable setting. A general reader should
understand the person's activities, records, and likely problems without
specialized training or outside research. Prefer common roles and recurring
life activities over niche occupations, rare equipment, technical procedures,
or institution-specific jargon. When a specialized term is unavoidable,
explain it immediately in plain language and do not make it the main challenge.

Return JSON only with these fields in order:
1. instance_id: copy the assigned ID.
2. instance_description: concrete setting, activities, records, and boundaries.
3. personal_description: one protagonist with an ordinary full name. The full
   name is the database user_id and must not reuse a listed protagonist name.
4. commands: initialization commands.

Commands must first create 2-3 SQLite tables (at most 5 columns each), including
one user-profile table; create 2-3 empty .txt Files for reusable guidance; then
INSERT exactly one profile row and call wallet.setup_wallet once. Include no
Scenario records or personal-App activity.

Use these exact lowercase shapes:
- {"app":"database","function":"execute","parameter":"CREATE TABLE ...;","description":"..."}
- {"app":"files","function":"create_file","parameter":{"name":"guide.txt"},"description":"..."}
- {"app":"database","function":"execute","parameter":"INSERT INTO ...;"}
- {"app":"wallet","function":"setup_wallet","parameter":{"currency":"USD","balance":1000,"account_type":"personal"}}

Do not imitate Email, Calendar, Messaging, Contacts, Files, or Wallet in the
database. Make this Instance materially different from existing Instances in
the same Domain. A different name, institution, location, or wording is not
enough: vary the protagonist's concrete role and responsibilities, central
activities, recurring workflows, and record/reference design.

Assigned Domain:
<<DOMAIN_SPEC>>

Assigned Instance ID: <<INSTANCE_ID>>

Existing Instances in this Domain:
<<EXISTING_INSTANCES>>

Protagonist names already used elsewhere in the benchmark:
<<USED_PROTAGONIST_NAMES>>

App reference:
<<APP_REFERENCE>>"""


TRIGGER_DEFINITIONS = {
    "dependency": "A prerequisite is missing for a future activity, but no active attempt is blocked yet.",
    "blockage": "The protagonist has begun executing a specific activity, but that same active attempt has actually stopped and cannot continue without help.",
    "risk": "No loss has happened yet, but inaction could plausibly cause a future negative outcome.",
    "opportunity": "Action could create an optional benefit; ignoring it causes no failure or loss.",
}


SCENARIO_CREATOR_PROMPT = r"""You are the Scenario Creator for a proactive-agent benchmark.

Create one <<TRIGGER_TYPE>> Scenario and any initial data it needs.
Definition: <<TRIGGER_DEFINITION>>

Return JSON only with exactly these fields in order:
1. scenario_description
2. events: event1...eventN
3. ini_data_commands

The program adds scenario_id, trigger_type, and reference_time. Treat the
assigned reference date as the Scenario's current date and keep all timestamps
chronologically consistent with it. Return exactly <<EVENT_COUNT>>
chronological Events. Every Event contains exactly app, function, and parameter;
never output oracle. At least one meaningful non-final checkpoint must still be
too early for intervention, while a later checkpoint makes the proactive need
sufficiently observable. Events must not contain the tested agent's response or
resolve the need.

Do not default to making the final Event the first actionable point. When
plausible, make the need sufficiently observable at a non-final Event, then use
later Events to add urgency, options, or confirmation without changing the fact
that intervention was already justified. Vary this timing across the accepted
Scenarios rather than repeating one Event pattern.

The proactive intervention must materially help the protagonist. Design a need
that benefits from connecting information across Events or shared state,
coordinating people or schedules, preparing a concrete next step, or safely
carrying out an action. Avoid cases where one explicit message already states
the whole problem and solution, the normal process is already resolving it, or
the Agent can only repeat a reminder. For dependency, blockage, and risk,
timely help should protect a real goal and preferably be necessary. Opportunity
must remain optional, but its benefit must be concrete and worthwhile.
The proposed intervention must occur soon enough to prevent or relieve the
stated consequence. Do not claim immediate urgency when every available remedy
comes too late. If a full fix must wait, include a useful and safe interim
action for the period before it becomes available.

For blockage, prove through an Event that the target activity has started in
practice, then show that the same active attempt cannot continue. Preparing,
scheduling, enrolling, or arranging prerequisites for a future activity is not
an active attempt. A future appointment conflict, missing future requirement,
or inability to attend later is dependency or risk, not blockage.

ini_data_commands is a list of app/function/parameter commands for facts that
exist before event1. It may be empty when the shared state and Events are already
sufficient. Otherwise create every hidden fact and Event prerequisite needed to
understand the situation. Use only existing SQLite tables and columns. Use
personal Apps for communications, appointments, contacts, and payments. Do not
recreate the protagonist, reset the Wallet, pre-apply an Event, cross a later
trigger threshold early, or introduce another proactive need. Do not add noise;
a separate Noise Creator runs after all eight Scenarios.

Every app value must be exactly one of these lowercase strings: database, files,
email, calendar, messaging, contacts, wallet. Do not use class names such as
Database or Email. Use exact functions from the App reference. Do not use query functions as Events,
invent App IDs, or use Web as an Event. Personal-App Events must be visible to
the protagonist. All identifiers and natural keys must remain unique.

Use plain language that a general reader can judge without professional or
technical knowledge. Do not make obscure credentials, regulations, equipment,
or terminology essential to recognizing the need.

The eight Scenarios share accumulated initial data, but their Events run in
independent copies and are never each other's history. Make the new need,
objects, causal progression, and intervention materially different from all
accepted Scenarios. For every same-trigger Scenario, differ in at least three
of these four dimensions: affected workflow or goal, cause of the need, key
resource or record, and likely intervention. Changing only the object is not
enough; do not reuse the same central activity plus the same failure mechanism.
Across all eight Scenarios, cover multiple activity areas from the Instance
description. Prefer an activity area not yet covered by accepted Scenarios and
avoid concentrating the set on one central project or activity. Mentioning a
different course, tool, or location inside the same central project does not
make it a new area.
When retry feedback exists, correct every listed issue with a fresh Scenario
and fresh ini_data_commands.
If feedback reports duplication, abandon that causal template completely:
change the main Event types, how the need emerges, and the expected response.
In particular, do not repeatedly express opportunity as announcement →
eligibility → deadline → application; opportunities may instead arise from a
newly usable resource, schedule fit, collaboration, reuse, or optional saving.
Compare the benefit mechanism, not just names and settings: two temporary rooms,
tools, services, or time slots that become available and require reservation
are the same opportunity template. Abstract each opportunity as source of new
value → required coordination → benefit, and do not reuse that abstraction.
Changing one free or leftover item into another and matching it to a different
use is still the same reuse mechanism.

Before returning, silently verify chronological timestamps, App and SQL
arguments, availability of every fact needed for the proposed intervention,
trigger semantics, and the true earliest justified intervention Event.
Every external reply, personalized offer, or service appointment must have a
visible cause in earlier Events or initial state, such as a request, referral,
subscription, or clearly explained automatic notice. Do not let an organization
respond to contact that never occurred.

Assigned Scenario ID: <<SCENARIO_ID>>
Assigned reference date: <<REFERENCE_DATE>>
Protagonist: <<PERSONAL_DESCRIPTION>>

Current shared Instance state:
<<INSTANCE_CONTEXT>>

Accepted Scenarios with the same trigger type (primary duplicate check):
<<SAME_TRIGGER_SCENARIOS>>

Accepted Scenarios with other trigger types (consistency check only):
<<OTHER_SCENARIOS>>

Retry feedback:
<<RETRY_FEEDBACK>>

App reference:
<<APP_REFERENCE>>"""


REVIEWER_PROMPT = r"""You are the single semantic Reviewer for a proactive-agent benchmark.

The deterministic Validator has already checked JSON shape, App/SQL execution,
and isolated Event replay. Review all accepted Scenarios plus the candidate.
The candidate is not committed. Its ini_data_commands would join the shared
initial state; every Scenario's Events then evolves independently from it.

Return JSON only with exactly:
{"valid": boolean, "issues": [concise concrete strings]}

Reject only material defects:
- wrong trigger semantics;
- blockage does not prove that the protagonist began executing the target
  activity before that same attempt stalled; preparation, a future schedule
  conflict, a missing future prerequisite, or inability to attend later is not
  blockage;
- the Instance or candidate depends on unexplained specialist knowledge rather
  than an ordinary, generally understandable life situation;
- initial data contradicts the shared world, changes identity, pre-applies an
  Event, crosses a later threshold, or creates another proactive need;
- base reference tables, guide Files, and personal Apps are intentionally sparse
  or empty. Adding stable pre-event rows, guide content, contacts, conversations,
  appointments, or other Scenario prerequisites is allowed and is not itself a
  contradiction; absence from the base only means it was not initialized yet;
- required initial evidence or an Event prerequisite is missing;
- ini_data_commands are empty even though the shared state and Events are not
  sufficient; empty commands are otherwise valid;
- the sequence lacks a meaningful pre-intervention checkpoint or a clear
  earliest justified intervention point;
- the final Event is used as the first actionable point by habit or artificial
  information withholding; non-final intervention points should appear across
  the set when causally plausible;
- intervention would add little value because one message already gives the
  whole problem and solution, an existing process is already resolving it, or
  the Agent could only repeat an obvious reminder;
- the proposed intervention cannot prevent or relieve the stated consequence
  in time; when the full fix is later, no useful safe interim action exists;
- Events use future evidence, resolve the need, include the tested agent's
  response, or are not plausible and observable to the protagonist;
- an external reply, personalized offer, or appointment lacks a prior request,
  referral, subscription, or other explicit causal source;
- the candidate duplicates or contradicts an accepted Scenario;
- a same-trigger pair has essentially the same causal progression and expected
  intervention even when objects are renamed; require differences in at least
  three of: affected workflow or goal, cause, key resource or record, and likely
  intervention, and reject reuse of the same central activity plus failure
  mechanism;
- two opportunities reuse the same benefit mechanism, such as different
  limited resources or time slots becoming available for reservation, despite
  different objects or activity areas;
- two opportunities have the same source-of-value → coordination → benefit
  structure; in particular, different free or leftover items matched to
  different uses still share one reuse mechanism;
- the candidate unnecessarily concentrates the set on an already well-covered
  central project or activity when the Instance offers unused activity areas;
- any accepted Scenario becomes inconsistent with the proposed shared state.

Trigger definitions: dependency=future prerequisite missing; blockage=active
attempt actually stalled; risk=plausible future loss; opportunity=optional
benefit with no loss if ignored. During intermediate rounds, do not complain
that future trigger types or eight total Scenarios are not present yet. Do not
reject style preferences or safe additions of genuinely new objects.

Expected candidate trigger: <<EXPECTED_TRIGGER>>
Domain: <<DOMAIN_SPEC>>
Instance: <<INSTANCE_DESCRIPTION>>

Accepted Scenarios:
<<ACCEPTED_SCENARIOS>>

Candidate Scenario, including ini_data_commands:
<<CANDIDATE_SCENARIO>>

Shared state before the candidate:
<<BASE_STATE>>"""


NOISE_CREATOR_PROMPT = r"""You are the Noise Creator for a proactive-agent benchmark.

All eight Scenarios are complete. Add realistic but irrelevant background data
to the shared initial state. Return JSON only with exactly one field:
{"noise_commands": [...]}

Use 12-20 app/function/parameter commands. Add a small mix of Domain-coherent
completed database records and one harmless TXT note, plus ordinary background
Email, past Calendar activity, resolved Messaging, Contacts, and completed
Wallet activity. Use only existing tables and columns and exact App signatures.
Before create_message, its exact sender-and-recipient conversation must already
exist in the shared state or be created earlier with create_conversation.

Noise must not mention, duplicate, modify, complete, contradict, or provide
evidence for any accepted Scenario. It must not create a dependency, blockage,
risk, opportunity, deadline, unresolved request, pending transaction, future
obligation, warning, or tested-agent response. Never modify or delete existing
objects, insert another user/profile/owner, reinitialize the Wallet, or reuse an
existing natural key. Use completed or clearly inert past activity and keep
payments affordable.

Protagonist: <<PERSONAL_DESCRIPTION>>

Final shared state before noise:
<<INSTANCE_CONTEXT>>

Accepted Scenarios:
<<ACCEPTED_SCENARIOS>>

App reference:
<<APP_REFERENCE>>"""


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
    app_reference,
    benchmark_year,
):
    return _fill(
        INSTANCE_CREATOR_PROMPT,
        {
            "<<DOMAIN_SPEC>>": domain_spec,
            "<<BENCHMARK_YEAR>>": str(benchmark_year),
            "<<INSTANCE_ID>>": instance_id,
            "<<EXISTING_INSTANCES>>": existing_instances,
            "<<USED_PROTAGONIST_NAMES>>": used_protagonist_names,
            "<<APP_REFERENCE>>": app_reference,
        },
    )


def build_scenario_creator_prompt(
    trigger_type,
    scenario_id,
    instance_context,
    personal_description,
    same_trigger_scenarios,
    other_scenarios,
    app_reference,
    reference_date,
    retry_feedback=None,
    target_event_count=5,
):
    return _fill(
        SCENARIO_CREATOR_PROMPT,
        {
            "<<TRIGGER_TYPE>>": trigger_type,
            "<<TRIGGER_DEFINITION>>": TRIGGER_DEFINITIONS[trigger_type],
            "<<SCENARIO_ID>>": scenario_id,
            "<<REFERENCE_DATE>>": reference_date,
            "<<EVENT_COUNT>>": str(target_event_count),
            "<<PERSONAL_DESCRIPTION>>": personal_description,
            "<<INSTANCE_CONTEXT>>": instance_context,
            "<<SAME_TRIGGER_SCENARIOS>>": same_trigger_scenarios,
            "<<OTHER_SCENARIOS>>": other_scenarios,
            "<<RETRY_FEEDBACK>>": retry_feedback or [],
            "<<APP_REFERENCE>>": app_reference,
        },
    )


def build_reviewer_prompt(
    domain_spec,
    instance_description,
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
            "<<INSTANCE_DESCRIPTION>>": instance_description,
            "<<BASE_STATE>>": base_state,
            "<<ACCEPTED_SCENARIOS>>": accepted_scenarios,
            "<<CANDIDATE_SCENARIO>>": candidate_scenario,
        },
    )


def build_noise_creator_prompt(
    instance_context,
    personal_description,
    accepted_scenarios,
    app_reference,
):
    return _fill(
        NOISE_CREATOR_PROMPT,
        {
            "<<INSTANCE_CONTEXT>>": instance_context,
            "<<PERSONAL_DESCRIPTION>>": personal_description,
            "<<ACCEPTED_SCENARIOS>>": accepted_scenarios,
            "<<APP_REFERENCE>>": app_reference,
        },
    )
