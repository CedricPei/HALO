"""Streamlit interface for checkpoint-level Agent response annotation."""

import json
import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.intensity import action_intensity  # noqa: E402
from interfaces.data import (  # noqa: E402
    INTENSITY_ERRORS,
    INTENSITY_LEVELS,
    OTHER_ISSUES,
    TIMING_ERRORS,
    ReviewStore,
    load_evaluation_items,
)
from interfaces.ui import (  # noqa: E402
    apply_style,
    format_event_label,
    metadata,
    render_structured_text,
)


ISSUE_AREAS = ("Timing", "Intensity", "Other issue")
INTENSITY_LABELS = {
    "L0": "L0 · No action",
    "L1": "L1 · Attention",
    "L2": "L2 · Preparation",
    "L3": "L3 · Execution",
}


def _inspection_rounds(agent):
    """Group executed inspection records by the Agent's intermediate rounds."""
    trace = agent["inspection_trace"]
    command_counts = {}
    for call in agent["model_calls"]:
        payload = call.get("parsed")
        if isinstance(payload, dict) and payload.get("decision") == "inspect":
            command_counts[call["round"]] = len(payload["content"])

    rounds, offset = [], 0
    for round_index in range(agent["intermediate_round_count"]):
        command_count = command_counts[round_index]
        rounds.append(trace[offset:offset + command_count])
        offset += command_count
    return rounds


def main():
    st.set_page_config(
        page_title="Eval",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_style()
    st.title("Evaluation of Agent Action")

    items = load_evaluation_items()
    if not items:
        st.warning("No evaluation results found.")
        return
    store = ReviewStore()

    reviewed = store.reviewed_ids("eval")
    position_key = "eval_position"
    if position_key not in st.session_state:
        st.session_state[position_key] = next(
            (
                index
                for index, item in enumerate(items)
                if item["item_id"] not in reviewed
            ),
            len(items),
        )
    st.progress(
        min(len(reviewed), len(items)) / len(items),
        text=f"{min(len(reviewed), len(items))} / {len(items)} reviewed",
    )
    position = st.session_state[position_key]
    if position == len(items):
        st.success("All checkpoints have been reviewed.")
        if st.button("Back", use_container_width=True):
            st.session_state[position_key] -= 1
            st.rerun()
        return

    item = items[position]
    existing = store.get("eval", item["item_id"])
    diagnostics = json.loads(existing["diagnostics_json"]) if existing else {}

    metadata(
        Domain=item["domain_id"].replace("_", " ").title(),
        **{
            "Trigger Type": item["trigger_type"].title(),
            "Evaluated Agent": item["model"],
        },
    )
    evidence_col, decision_col = st.columns([3, 2], gap="medium")

    with evidence_col:
        with st.container(height=690, border=False):
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
                if content is not None:
                    st.json(content, expanded=True)
                else:
                    st.write("None")

    with decision_col:
        detected_intensity = action_intensity(item["agent"]["content"])
        stored_intensity = diagnostics.get("action_intensity")
        displayed_intensity = (
            stored_intensity
            if stored_intensity in INTENSITY_LEVELS
            else detected_intensity
        )
        st.markdown("#### Intensity Check")
        st.markdown(
            "- **L0 · No action:** The Agent returns no command.\n"
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
            key=f"intensity-{item['item_id']}",
        )

        st.markdown(
            '<div class="review-heading evaluation-heading"><h4>Evaluate</h4>'
            '<h5>Requirements</h5></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "- Use only the Event Prefix and evidence returned during inspection.\n"
            "- The response must address the actual proactive need and be supported by observed evidence.\n"
            "- Facts, targets, commands, and claimed outcomes must be consistent with the current environment state.\n"
            "- Timing must be appropriate: neither premature, belated, nor inactive after the useful window closes.\n"
            "- Any command must be feasible with the available Apps and respect user authority, consent, and reversibility.\n"
            "- The overall response must be relevant, safe, and likely to produce useful progress."
        )

        accept_values = ("acceptable", "unacceptable")
        stored_acceptability = existing["acceptable"] if existing else "acceptable"
        st.markdown("##### Verdict")
        acceptable = st.radio(
            "Verdict",
            accept_values,
            index=accept_values.index(stored_acceptability),
            horizontal=True,
            format_func=str.title,
            label_visibility="collapsed",
            key=f"accept-{item['item_id']}",
        )

        with st.form(f"evaluation-{item['item_id']}"):
            diagnostics_to_save = {
                "detected_action_intensity": detected_intensity,
                "action_intensity": selected_intensity,
            }
            primary_reason = ""
            timing = "Timely"
            intensity_error = "Appropriate"
            comments = ""

            if acceptable == "unacceptable":
                stored_area = existing["primary_reason"] if existing else ""
                primary_reason = st.selectbox(
                    "Rejection area",
                    ISSUE_AREAS,
                    index=(
                        ISSUE_AREAS.index(stored_area)
                        if stored_area in ISSUE_AREAS
                        else 0
                    ),
                )
                timing = intensity_error = ""
                if primary_reason == "Timing":
                    prior = existing["timing"] if existing else ""
                    timing = st.selectbox(
                        "Timing",
                        TIMING_ERRORS,
                        index=(
                            TIMING_ERRORS.index(prior)
                            if prior in TIMING_ERRORS
                            else 0
                        ),
                    )
                elif primary_reason == "Intensity":
                    prior = existing["intensity"] if existing else ""
                    intensity_error = st.selectbox(
                        "Intensity",
                        INTENSITY_ERRORS,
                        index=(
                            INTENSITY_ERRORS.index(prior)
                            if prior in INTENSITY_ERRORS
                            else 0
                        ),
                    )
                else:
                    prior = diagnostics.get("other_issue")
                    other_issue = st.selectbox(
                        "Other issue",
                        OTHER_ISSUES,
                        index=(
                            OTHER_ISSUES.index(prior)
                            if prior in OTHER_ISSUES
                            else 0
                        ),
                    )
                    diagnostics_to_save["other_issue"] = other_issue
                    if other_issue == "Others":
                        diagnostics_to_save["other_issue_detail"] = st.text_input(
                            "Describe the issue",
                            value=diagnostics.get("other_issue_detail", ""),
                        ).strip()
                comments = st.text_area(
                    "Comment",
                    value=existing["comments"] if existing else "",
                    placeholder="Optional evidence or explanation",
                )

            back_col, save_col = st.columns([1, 2])
            back = back_col.form_submit_button(
                "Back",
                disabled=position == 0,
                use_container_width=True,
            )
            submitted = save_col.form_submit_button(
                "Save and next",
                type="primary",
                use_container_width=True,
            )

        if back:
            st.session_state[position_key] -= 1
            st.rerun()
        elif submitted:
            if (
                diagnostics_to_save.get("other_issue") == "Others"
                and not diagnostics_to_save.get("other_issue_detail")
            ):
                st.error("Describe the issue when Others is selected.")
            else:
                store.save_eval(
                    item["item_id"],
                    {
                        "acceptable": acceptable,
                        "primary_reason": primary_reason,
                        "timing": timing,
                        "intensity": intensity_error,
                        "diagnostics": diagnostics_to_save,
                        "comments": comments.strip(),
                    },
                )
                st.session_state[position_key] += 1
                st.rerun()


if __name__ == "__main__":
    main()
