"""Annotator dashboard plus the two existing review workflows."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from annotation_app.storage import AnnotationStore
from evaluation.intensity import INTENSITY_LEVELS, action_intensity
from annotation_app.ui import format_event_label, metadata, render_structured_text


TIMING_ERRORS = ("Premature", "Belated")
TIMING_CHECKS = ("Premature", "Appropriate", "Belated")
TIMING_STORAGE = {
    "Premature": "Premature",
    "Appropriate": "Timely",
    "Belated": "Belated",
}
INACTION_ERROR = "Inaction"
INTENSITY_ERRORS = ("Insufficient", "Excessive")
INTENSITY_CHECKS = ("Insufficient", "Appropriate", "Excessive")
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


def _render_review_progress(
    items: list[dict],
    position: int,
    reviewed_count: int,
    position_key: str,
    review_type: str,
    user_id: str,
) -> None:
    """Render compact review status, numeric navigation, and the progress bar."""
    current_position = position + 1 if position < len(items) else "—"
    current_col, reviewed_col, jump_col = st.columns(
        [1.2, 1.2, 0.9], gap="medium", vertical_alignment="bottom"
    )
    current_col.markdown(
        '<div class="review-stat"><span>Current</span>'
        f"<strong>{current_position} / {len(items)}</strong></div>",
        unsafe_allow_html=True,
    )
    reviewed_col.markdown(
        '<div class="review-stat"><span>Reviewed</span>'
        f"<strong>{reviewed_count} / {len(items)}</strong></div>",
        unsafe_allow_html=True,
    )
    with jump_col:
        with st.form(f"cloud-{review_type}-id-jump-{user_id}", border=False):
            label_col, input_col, button_col = st.columns(
                [0.75, 0.45, 0.42], gap="small", vertical_alignment="bottom"
            )
            label_col.markdown(
                '<div class="id-jump-marker">Go to</div>',
                unsafe_allow_html=True,
            )
            target_id = input_col.text_input(
                "Go to Item ID",
                placeholder="ID",
                label_visibility="collapsed",
                key=f"cloud-{review_type}-jump-target-{user_id}",
            )
            jump = button_col.form_submit_button("Go", use_container_width=True)

    if jump:
        target_id = target_id.strip()
        if not target_id.isdecimal():
            st.error("Enter a numeric Item ID.")
        else:
            positions = {
                str(item["display_id"]): index for index, item in enumerate(items)
            }
            if target_id not in positions:
                st.error(f"Item ID {target_id} is not assigned to this account.")
            else:
                st.session_state[position_key] = positions[target_id]
                st.rerun()

    st.markdown(
        '<div class="review-progress-gap" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )
    st.progress(reviewed_count / len(items))


def _enable_copy_protection() -> None:
    """Disable ordinary selection/copy actions while an annotator reviews a task."""
    st.markdown(
        '<span class="copy-protection-marker" aria-hidden="true"></span>',
        unsafe_allow_html=True,
    )
    components.html(
        """
        <script>
          try {
            const doc = window.parent.document;
            const markerSelector = ".copy-protection-marker";
            if (doc.documentElement.dataset.proactCopyGuard !== "enabled") {
              const blockWhenReviewing = (event) => {
                if (doc.querySelector(markerSelector)) {
                  event.preventDefault();
                }
              };
              doc.addEventListener("copy", blockWhenReviewing, true);
              doc.addEventListener("cut", blockWhenReviewing, true);
              doc.addEventListener("contextmenu", blockWhenReviewing, true);
              doc.addEventListener("dragstart", blockWhenReviewing, true);
              doc.documentElement.dataset.proactCopyGuard = "enabled";
            }
          } catch (_) {
            // CSS selection blocking remains active if iframe access is restricted.
          }
        </script>
        """,
        height=0,
        width=0,
    )


def _ordered_assigned_items(
    store: AnnotationStore,
    user_id: str,
    review_type: str,
    all_items: list[dict],
) -> tuple[list[dict], int]:
    by_id = {
        item["item_id"]: {**item, "display_id": display_id}
        for display_id, item in enumerate(all_items, start=1)
    }
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


def render_audit(
    store: AnnotationStore,
    user: dict,
    all_items: list[dict],
) -> None:
    _enable_copy_protection()
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
    reviewed_count = min(len(reviewed), len(items))
    _render_review_progress(
        items,
        position,
        reviewed_count,
        position_key,
        "audit",
        user_id,
    )
    if position == len(items):
        st.success("All assigned Scenarios have been reviewed.")
        if st.button("Back", use_container_width=True, key="audit-complete-back"):
            st.session_state[position_key] -= 1
            st.rerun()
        return

    item = items[position]
    saved = store.get_annotation(user_id, "audit", item["item_id"])
    existing = saved["payload"] if saved else {}

    metadata(
        **{"Item ID": item["display_id"]},
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
        stored_verdict = existing.get("verdict")
        st.markdown("##### Verdict")
        verdict = st.radio(
            "Verdict",
            verdict_values,
            index=(
                verdict_values.index(stored_verdict)
                if stored_verdict in verdict_values
                else None
            ),
            horizontal=True,
            format_func=lambda value: value.replace("_", " ").title(),
            label_visibility="collapsed",
            key=f"cloud-verdict-{user_id}-{item['item_id']}",
        )
        with st.form(f"cloud-construction-{user_id}-{item['item_id']}"):
            comment = st.text_area(
                "Comment (optional)",
                value=existing.get("comment", ""),
                placeholder="Add context that would help explain your judgment.",
                height=80,
            )
            back_col, save_col = st.columns([1, 2])
            back = back_col.form_submit_button(
                "Back", disabled=position == 0, use_container_width=True
            )
            submitted = save_col.form_submit_button(
                "Save and next",
                type="primary",
                use_container_width=True,
                disabled=verdict is None,
            )
        if back:
            st.session_state[position_key] -= 1
            st.rerun()
        elif submitted:
            if verdict is None:
                st.error("Select a verdict before continuing.")
            else:
                store.save_annotation(
                    user_id,
                    "audit",
                    item["item_id"],
                    {
                        **existing,
                        "verdict": verdict,
                        "comment": comment.strip(),
                    },
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


def _stored_timing_check(
    existing: dict,
    stored_dimensions: set[str],
    has_saved_unacceptable_review: bool,
) -> str | None:
    if not has_saved_unacceptable_review:
        return None
    prior = existing.get("timing", "")
    if prior in TIMING_ERRORS:
        return prior
    if prior == "Timely" or "Timing" not in stored_dimensions:
        return "Appropriate"
    return None


def _stored_intensity_check(
    existing: dict,
    stored_dimensions: set[str],
    has_saved_unacceptable_review: bool,
) -> str | None:
    if not has_saved_unacceptable_review:
        return None
    prior = existing.get("intensity", "")
    if prior in INTENSITY_ERRORS:
        return prior
    if prior == "Appropriate" or "Intensity" not in stored_dimensions:
        return "Appropriate"
    return None


def render_eval(
    store: AnnotationStore,
    user: dict,
    all_items: list[dict],
) -> None:
    _enable_copy_protection()
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
    reviewed_count = min(len(reviewed), len(items))
    _render_review_progress(
        items,
        position,
        reviewed_count,
        position_key,
        "eval",
        user_id,
    )
    if position == len(items):
        st.success("All assigned checkpoints have been reviewed.")
        if st.button("Back", use_container_width=True, key="eval-complete-back"):
            st.session_state[position_key] -= 1
            st.rerun()
        return

    item = items[position]
    saved = store.get_annotation(user_id, "eval", item["item_id"])
    existing = saved["payload"] if saved else {}
    diagnostics = existing.get("diagnostics", {})

    metadata(
        **{"Item ID": item["display_id"]},
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
                "- **L1 · Attention:** The command presents a notification, reminder, or flagged item for the user’s attention.\n"
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
            '<li>The intervention must have appropriate intensity and remain safe, authorized, and likely to produce useful progress.</li>'
            '<li>A user notification needs grounded, useful information or a concrete next step; generic advice or restating Events alone is insufficient.</li>'
            '<li>A notification does not establish user approval, a reply, or completion of proposed operations. A useful L1 message can be sufficient; higher intensity is not inherently better.</li></ul></div>',
            unsafe_allow_html=True,
        )

        verdict_values = ("acceptable", "unacceptable")
        stored_verdict = existing.get("verdict", existing.get("acceptable"))
        st.markdown("##### Verdict")
        verdict = st.radio(
            "Verdict",
            verdict_values,
            index=(
                verdict_values.index(stored_verdict)
                if stored_verdict in verdict_values
                else None
            ),
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
        diagnostics_complete = True
        if verdict == "unacceptable" and selected_intensity == "L0":
            dimensions = ["Timing"]
            timing = INACTION_ERROR
            intensity_error = ""
            diagnostics_to_save["rejection_dimensions"] = dimensions
        elif verdict == "unacceptable":
            stored_dimensions = _stored_dimensions(existing, diagnostics)
            st.markdown(
                '<div class="rejection-heading">Rejection Reasons</div>',
                unsafe_allow_html=True,
            )
            timing_check = _stored_timing_check(
                existing,
                stored_dimensions,
                stored_verdict == "unacceptable",
            )
            timing_check = st.radio(
                "Regardless of the intervention intensity, was this an "
                "appropriate time to intervene?",
                TIMING_CHECKS,
                index=(
                    TIMING_CHECKS.index(timing_check)
                    if timing_check in TIMING_CHECKS
                    else None
                ),
                horizontal=True,
                key=f"cloud-timing-check-{user_id}-{item['item_id']}",
            )
            intensity_check = _stored_intensity_check(
                existing,
                stored_dimensions,
                stored_verdict == "unacceptable",
            )
            intensity_check = st.radio(
                "At this point, was the intervention intensity appropriate?",
                INTENSITY_CHECKS,
                index=(
                    INTENSITY_CHECKS.index(intensity_check)
                    if intensity_check in INTENSITY_CHECKS
                    else None
                ),
                horizontal=True,
                key=f"cloud-intensity-appropriateness-{user_id}-{item['item_id']}",
            )
            other_issue_selected = st.checkbox(
                "Other issues",
                value="Other issues" in stored_dimensions,
                key=f"cloud-dimension-Other issues-{user_id}-{item['item_id']}",
            )

            timing = TIMING_STORAGE.get(timing_check, "")
            intensity_error = intensity_check or ""
            if timing_check in TIMING_ERRORS:
                dimensions.append("Timing")
            if intensity_check in INTENSITY_ERRORS:
                dimensions.append("Intensity")
            if other_issue_selected:
                dimensions.append("Other issues")
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
            checks_answered = (
                timing_check is not None and intensity_check is not None
            )
            has_rejection_reason = bool(dimensions)
            other_detail_complete = not (
                other_issue_selected
                and other_issue == "Others"
                and not other_issue_detail
            )
            diagnostics_complete = (
                checks_answered
                and has_rejection_reason
                and other_detail_complete
            )
            if checks_answered and not has_rejection_reason:
                st.caption(
                    "Timing and intensity are appropriate. Select an Other Issue "
                    "or change the Verdict to Acceptable."
                )
            elif not other_detail_complete:
                st.caption("Describe the issue when Others is selected.")
            st.markdown(
                '<div class="eval-actions-spacer"></div>',
                unsafe_allow_html=True,
            )

        with st.form(f"cloud-evaluation-{user_id}-{item['item_id']}"):
            comments = st.text_area(
                "Comment (optional)",
                value=existing.get("comments", ""),
                placeholder="Add context that would help explain your judgment.",
                height=56,
            )
            back_col, save_col = st.columns([1, 2])
            back = back_col.form_submit_button(
                "Back", disabled=position == 0, use_container_width=True
            )
            submitted = save_col.form_submit_button(
                "Save and next",
                type="primary",
                use_container_width=True,
                disabled=verdict is None or not diagnostics_complete,
            )
        if back:
            st.session_state[position_key] -= 1
            st.rerun()
        elif submitted:
            if verdict is None:
                st.error("Select a verdict before continuing.")
            elif not diagnostics_complete:
                st.error("Complete the rejection checks before continuing.")
            else:
                primary_reason = ", ".join(dimensions)
                updated = {
                    **existing,
                    "verdict": verdict,
                    "primary_reason": primary_reason,
                    "timing": timing,
                    "intensity": intensity_error,
                    "diagnostics": diagnostics_to_save,
                    "comments": comments.strip(),
                }
                updated.pop("acceptable", None)
                store.save_annotation(
                    user_id,
                    "eval",
                    item["item_id"],
                    updated,
                )
                st.session_state[position_key] += 1
                st.rerun()
