"""Annotator dashboard plus the two existing review workflows."""

from __future__ import annotations

import streamlit as st

from annotation_app.storage import AnnotationStore
from evaluation.intensity import INTENSITY_LEVELS, action_intensity
from interfaces.ui import format_event_label, metadata, render_structured_text


REJECTION_DIMENSIONS = ("Timing", "Intensity", "Other issues")
TIMING_ERRORS = ("Premature", "Belated")
INACTION_ERROR = "Inaction"
INTENSITY_ERRORS = ("Insufficient", "Excessive")
OTHER_ISSUES = (
    "Unsupported Evidence",
    "Irrelevant Action",
    "Factual Contradiction",
    "Others",
)
INTENSITY_LABELS = {
    "L0": "L0 · No action or retrieval",
    "L1": "L1 · Attention",
    "L2": "L2 · Preparation",
    "L3": "L3 · Execution",
}


def _ordered_assigned_items(
    store: AnnotationStore,
    user_id: str,
    review_type: str,
    all_items: list[dict],
) -> tuple[list[dict], int]:
    by_id = {item["item_id"]: item for item in all_items}
    assigned_ids = store.assignment_ids(user_id, review_type)
    items = [by_id[item_id] for item_id in assigned_ids if item_id in by_id]
    return items, len(assigned_ids) - len(items)


def _initial_position(items: list[dict], reviewed: set[str]) -> int:
    return next(
        (
            index
            for index, item in enumerate(items)
            if item["item_id"] not in reviewed
        ),
        len(items),
    )


def render_dashboard(store: AnnotationStore, user: dict) -> None:
    st.title("Annotation Workspace")
    st.caption(f"Signed in as {user['display_name']} · {user['user_id']}")
    progress = store.progress_for_user(user["user_id"])
    columns = st.columns(2)
    for column, review_type, label in zip(
        columns,
        ("audit", "eval"),
        ("Benchmark Audit", "Agent Evaluation"),
    ):
        values = progress[review_type]
        assigned = values["assigned"]
        submitted = values["submitted"]
        percentage = int(100 * submitted / assigned) if assigned else 0
        with column:
            st.markdown(f"#### {label}")
            st.metric("Completed", f"{submitted} / {assigned}")
            st.progress(percentage / 100, text=f"{percentage}%")
    if not any(values["assigned"] for values in progress.values()):
        st.info("No tasks have been assigned to this account yet.")
    else:
        st.markdown(
            "Use the navigation on the left to continue your assigned work. "
            "Every save creates a new revision; original benchmark data is read-only."
        )


def render_audit(
    store: AnnotationStore,
    user: dict,
    all_items: list[dict],
) -> None:
    user_id = user["user_id"]
    items, missing_count = _ordered_assigned_items(
        store, user_id, "audit", all_items
    )
    st.title("Benchmark Audit")
    if missing_count:
        st.warning(
            f"{missing_count} assigned item(s) are absent from this source version."
        )
    if not items:
        st.info("No Benchmark Audit tasks are assigned to this account.")
        return

    reviewed = store.reviewed_ids(user_id, "audit")
    position_key = f"cloud_audit_position_{user_id}"
    if position_key not in st.session_state:
        st.session_state[position_key] = _initial_position(items, reviewed)
    position = min(max(st.session_state[position_key], 0), len(items))
    st.session_state[position_key] = position
    st.progress(
        min(len(reviewed), len(items)) / len(items),
        text=f"{min(len(reviewed), len(items))} / {len(items)} reviewed",
    )
    if position == len(items):
        st.success("All assigned Scenarios have been reviewed.")
        if st.button("Back", use_container_width=True, key="audit-complete-back"):
            st.session_state[position_key] -= 1
            st.rerun()
        return

    item = items[position]
    latest = store.latest_annotation(user_id, "audit", item["item_id"])
    existing = latest["payload"] if latest else {}
    if latest:
        st.caption(
            f"Saved version {latest['version']} · Editing creates a new revision."
        )

    metadata(
        Domain=item["domain_id"].replace("_", " ").title(),
        **{"Trigger Type": item["trigger_type"].title()},
    )
    evidence_col, decision_col = st.columns([3, 2], gap="medium")

    with evidence_col:
        with st.container(height=690, border=False, key="evidence-pane"):
            st.markdown("#### Instance Overview")
            render_structured_text(item["instance"]["instance_description"])
            st.markdown("#### User Profile")
            render_structured_text(item["instance"]["personal_description"])
            st.markdown("#### Scenario")
            render_structured_text(item["scenario"]["scenario_description"])
            st.markdown("#### Events")
            events = list(item["scenario"]["events"].items())
            for index, (event_id, event) in enumerate(events):
                with st.expander(
                    format_event_label(event_id, event),
                    expanded=index == len(events) - 1,
                ):
                    st.caption(f"{event['app']}.{event['function']}")
                    st.json(event["parameter"], expanded=True)
            with st.expander("Supporting context"):
                tabs = st.tabs(["Database", "Files", "Apps"])
                tabs[0].json(item["state"]["database"], expanded=False)
                tabs[1].json(item["state"]["files"], expanded=False)
                tabs[2].json(item["state"]["apps"], expanded=False)

    with decision_col:
        st.markdown(
            '<div class="review-heading"><h4>Audit</h4>'
            '<h5>Requirements</h5></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="requirement-group"><strong>Timing</strong>'
            '<ul><li>One proactive need matches the assigned Trigger Type and develops through a causal, judgeable Event sequence.</li>'
            '<li>One prefix makes intervention unacceptably premature; a later prefix makes acting already too late.</li></ul></div>'
            '<div class="requirement-group"><strong>Evidence</strong>'
            '<ul><li>A necessary database or File fact materially affects recognition, timing, or response.</li>'
            '<li>A useful response is feasible in the valid window, not already completed, and not directly requested.</li></ul></div>'
            '<div class="requirement-group"><strong>Coherence</strong>'
            '<ul><li>Apps, state changes, authority, and the likely response are relevant, consistent, safe, and proportionate.</li>'
            '<li>The Scenario is materially distinct from accepted designs rather than a surface variation.</li></ul></div>',
            unsafe_allow_html=True,
        )
        verdict_values = ("approve", "needs_revision", "reject")
        stored_verdict = existing.get("verdict", "approve")
        st.markdown("##### Verdict")
        verdict = st.radio(
            "Verdict",
            verdict_values,
            index=verdict_values.index(stored_verdict),
            horizontal=True,
            format_func=lambda value: value.replace("_", " ").title(),
            label_visibility="collapsed",
            key=f"cloud-verdict-{user_id}-{item['item_id']}",
        )
        with st.form(f"cloud-construction-{user_id}-{item['item_id']}"):
            if verdict == "needs_revision":
                comment = st.text_area(
                    "Comment",
                    value=existing.get("comment", ""),
                    placeholder="What should be revised?",
                )
            else:
                comment = ""
            back_col, save_col = st.columns([1, 2])
            back = back_col.form_submit_button(
                "Back", disabled=position == 0, use_container_width=True
            )
            submitted = save_col.form_submit_button(
                "Save and next", type="primary", use_container_width=True
            )
        if back:
            st.session_state[position_key] -= 1
            st.rerun()
        elif submitted:
            store.save_annotation(
                user_id,
                "audit",
                item["item_id"],
                {"verdict": verdict, "comment": comment.strip()},
            )
            st.session_state[position_key] += 1
            st.rerun()


def _inspection_rounds(agent: dict) -> list[list[dict]]:
    trace = agent.get("inspection_trace", [])
    command_counts = {}
    for call in agent.get("model_calls", []):
        payload = call.get("parsed")
        if isinstance(payload, dict) and payload.get("decision") == "inspect":
            command_counts[call["round"]] = len(payload["content"])
    rounds, offset = [], 0
    for round_index in range(agent.get("intermediate_round_count", 0)):
        command_count = command_counts.get(round_index, 0)
        rounds.append(trace[offset : offset + command_count])
        offset += command_count
    return rounds


def _stored_dimensions(existing: dict, diagnostics: dict) -> set[str]:
    dimensions = diagnostics.get("rejection_dimensions")
    if isinstance(dimensions, list):
        return set(dimensions)
    primary_reason = existing.get("primary_reason", "")
    return {
        "Other issues" if value == "Other issue" else value
        for value in primary_reason.split(", ")
        if value
    }


def render_eval(
    store: AnnotationStore,
    user: dict,
    all_items: list[dict],
) -> None:
    user_id = user["user_id"]
    items, missing_count = _ordered_assigned_items(store, user_id, "eval", all_items)
    st.title("Evaluation of Agent Action")
    if missing_count:
        st.warning(
            f"{missing_count} assigned item(s) are absent from this source version."
        )
    if not items:
        st.info("No Agent Evaluation tasks are assigned to this account.")
        return

    reviewed = store.reviewed_ids(user_id, "eval")
    position_key = f"cloud_eval_position_{user_id}"
    if position_key not in st.session_state:
        st.session_state[position_key] = _initial_position(items, reviewed)
    position = min(max(st.session_state[position_key], 0), len(items))
    st.session_state[position_key] = position
    st.progress(
        min(len(reviewed), len(items)) / len(items),
        text=f"{min(len(reviewed), len(items))} / {len(items)} reviewed",
    )
    if position == len(items):
        st.success("All assigned checkpoints have been reviewed.")
        if st.button("Back", use_container_width=True, key="eval-complete-back"):
            st.session_state[position_key] -= 1
            st.rerun()
        return

    item = items[position]
    latest = store.latest_annotation(user_id, "eval", item["item_id"])
    existing = latest["payload"] if latest else {}
    diagnostics = existing.get("diagnostics", {})
    if latest:
        st.caption(
            f"Saved version {latest['version']} · Editing creates a new revision."
        )

    metadata(
        Domain=item["domain_id"].replace("_", " ").title(),
        **{
            "Trigger Type": item["trigger_type"].title(),
            "Evaluated Agent": item["model"],
        },
    )
    evidence_col, decision_col = st.columns([3, 2], gap="medium")
    with evidence_col:
        with st.container(height=690, border=False, key="evidence-pane"):
            st.markdown("#### Instance Overview")
            render_structured_text(item["instance"]["instance_description"])
            st.markdown("#### User Profile")
            render_structured_text(item["instance"]["personal_description"])
            st.markdown("#### Scenario")
            render_structured_text(item["scenario_description"])
            st.markdown("#### Event Prefix")
            for index, event in enumerate(item["event_prefix"]):
                with st.expander(
                    format_event_label(event["event_id"], event),
                    expanded=index == len(item["event_prefix"]) - 1,
                ):
                    st.caption(f"{event['app']}.{event['function']}")
                    st.json(event["parameter"], expanded=True)
            st.markdown("#### Agent Action")
            content = item["agent"]["content"]
            inspection_rounds = _inspection_rounds(item["agent"])
            with st.container(border=True):
                for round_number, records in enumerate(inspection_rounds, start=1):
                    with st.expander(f"Intermediate round {round_number}"):
                        for record in records:
                            command = record["command"]
                            st.markdown(f"**{command['app']}.{command['function']}**")
                            st.json(
                                {
                                    "request": command["parameter"],
                                    "result": record["result"],
                                },
                                expanded=False,
                            )
                if inspection_rounds:
                    st.markdown("**Response**")
                st.json(content, expanded=True) if content is not None else st.write(
                    "None"
                )

    with decision_col:
        detected_intensity = action_intensity(item["agent"]["content"])
        stored_intensity = diagnostics.get("action_intensity")
        displayed_intensity = (
            stored_intensity
            if stored_intensity in INTENSITY_LEVELS
            else detected_intensity
        )
        if item["agent"]["content"] is None:
            selected_intensity = "L0"
        else:
            st.markdown("#### Intensity Check")
            st.markdown(
                "- **L0 · No action or retrieval:** The Agent returns no command or a retrieval-only response.\n"
                "- **L1 · Attention:** The command surfaces something for the user’s attention.\n"
                "- **L2 · Preparation:** The command prepares an unexecuted operation or maintains internal contact information.\n"
                "- **L3 · Execution:** The command changes an active schedule, communicates externally, or moves money."
            )
            st.markdown(
                '<div class="intensity-inference">Automatically inferred: '
                f"<strong>{INTENSITY_LABELS[detected_intensity]}</strong> from "
                "the final command.</div>",
                unsafe_allow_html=True,
            )
            selected_intensity = st.selectbox(
                "Intensity Check",
                INTENSITY_LEVELS,
                index=INTENSITY_LEVELS.index(displayed_intensity),
                format_func=INTENSITY_LABELS.get,
                label_visibility="collapsed",
                key=f"cloud-intensity-{user_id}-{item['item_id']}",
            )

        st.markdown(
            '<div class="review-heading evaluation-heading"><h4>Evaluate</h4>'
            '<h5>Requirements</h5></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="requirement-group"><strong>Evidence</strong>'
            '<ul><li>Judge only the Event Prefix and evidence returned during intermediate rounds; do not use future Events.</li>'
            '<li>The action must address the actual proactive need, and every material claim must be supported by the observed state.</li></ul></div>'
            '<div class="requirement-group"><strong>Timing</strong>'
            '<ul><li>Act only after the necessary conditions exist and while useful progress is still possible.</li>'
            '<li>Returning no action is unacceptable when a supported, useful intervention is available at this checkpoint.</li></ul></div>'
            '<div class="requirement-group"><strong>Action</strong>'
            '<ul><li>The target, command, parameters, and claimed outcome must be relevant, feasible, and consistent with the environment.</li>'
            '<li>The intervention must have appropriate intensity and remain safe, authorized, and likely to produce useful progress.</li></ul></div>',
            unsafe_allow_html=True,
        )

        accept_values = ("acceptable", "unacceptable")
        stored_acceptability = existing.get("acceptable", "acceptable")
        st.markdown("##### Verdict")
        acceptable = st.radio(
            "Verdict",
            accept_values,
            index=accept_values.index(stored_acceptability),
            horizontal=True,
            format_func=str.title,
            label_visibility="collapsed",
            key=f"cloud-accept-{user_id}-{item['item_id']}",
        )

        diagnostics_to_save = {
            "detected_action_intensity": detected_intensity,
            "action_intensity": selected_intensity,
        }
        dimensions = []
        timing = "Timely"
        intensity_error = "Appropriate"
        other_issue = ""
        other_issue_detail = ""
        if acceptable == "unacceptable" and selected_intensity == "L0":
            dimensions = ["Timing"]
            timing = INACTION_ERROR
            intensity_error = ""
            diagnostics_to_save["rejection_dimensions"] = dimensions
        elif acceptable == "unacceptable":
            stored_dimensions = _stored_dimensions(existing, diagnostics)
            st.markdown(
                '<div class="rejection-heading">Rejection Reasons</div>',
                unsafe_allow_html=True,
            )
            dimension_columns = st.columns(3)
            for column, dimension in zip(
                dimension_columns, REJECTION_DIMENSIONS
            ):
                if column.checkbox(
                    dimension,
                    value=dimension in stored_dimensions,
                    key=f"cloud-dimension-{dimension}-{user_id}-{item['item_id']}",
                ):
                    dimensions.append(dimension)
            timing = intensity_error = ""
            if "Timing" in dimensions:
                prior = existing.get("timing", "")
                timing = st.selectbox(
                    "Timing",
                    TIMING_ERRORS,
                    index=TIMING_ERRORS.index(prior) if prior in TIMING_ERRORS else 0,
                    key=f"cloud-timing-{user_id}-{item['item_id']}",
                )
            if "Intensity" in dimensions:
                prior = existing.get("intensity", "")
                intensity_error = st.selectbox(
                    "Intensity",
                    INTENSITY_ERRORS,
                    index=(
                        INTENSITY_ERRORS.index(prior)
                        if prior in INTENSITY_ERRORS
                        else 0
                    ),
                    key=f"cloud-intensity-error-{user_id}-{item['item_id']}",
                )
            if "Other issues" in dimensions:
                prior = diagnostics.get("other_issue")
                other_issue = st.selectbox(
                    "Other issues",
                    OTHER_ISSUES,
                    index=OTHER_ISSUES.index(prior) if prior in OTHER_ISSUES else 0,
                    key=f"cloud-other-issue-{user_id}-{item['item_id']}",
                )
                diagnostics_to_save["other_issue"] = other_issue
                if other_issue == "Others":
                    other_issue_detail = st.text_input(
                        "Describe the issue",
                        value=diagnostics.get("other_issue_detail", ""),
                        key=f"cloud-other-detail-{user_id}-{item['item_id']}",
                    ).strip()
                    diagnostics_to_save["other_issue_detail"] = other_issue_detail
            diagnostics_to_save["rejection_dimensions"] = dimensions
            st.markdown(
                '<div class="eval-actions-spacer"></div>',
                unsafe_allow_html=True,
            )

        with st.form(f"cloud-evaluation-{user_id}-{item['item_id']}"):
            back_col, save_col = st.columns([1, 2])
            back = back_col.form_submit_button(
                "Back", disabled=position == 0, use_container_width=True
            )
            submitted = save_col.form_submit_button(
                "Save and next", type="primary", use_container_width=True
            )
        if back:
            st.session_state[position_key] -= 1
            st.rerun()
        elif submitted:
            if (
                acceptable == "unacceptable"
                and selected_intensity != "L0"
                and not dimensions
            ):
                st.error("Select at least one rejection dimension.")
            elif other_issue == "Others" and not other_issue_detail:
                st.error("Describe the issue when Others is selected.")
            else:
                primary_reason = ", ".join(dimensions)
                store.save_annotation(
                    user_id,
                    "eval",
                    item["item_id"],
                    {
                        "acceptable": acceptable,
                        "primary_reason": primary_reason,
                        "timing": timing,
                        "intensity": intensity_error,
                        "diagnostics": diagnostics_to_save,
                        "comments": "",
                    },
                )
                st.session_state[position_key] += 1
                st.rerun()
