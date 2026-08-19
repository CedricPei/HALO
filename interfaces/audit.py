"""Streamlit interface for post-construction Scenario audit."""

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from interfaces.data import ReviewStore, load_construction_items  # noqa: E402
from interfaces.ui import (  # noqa: E402
    apply_style,
    format_event_label,
    metadata,
    render_structured_text,
)


def main():
    st.set_page_config(
        page_title="Audit",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_style()
    st.title("Benchmark Audit")

    items = load_construction_items()
    if not items:
        st.warning("No accepted Scenarios were found.")
        return
    store = ReviewStore()

    reviewed = store.reviewed_ids("audit")
    position_key = "audit_position"
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
        st.success("All Scenarios have been reviewed.")
        if st.button("Back", use_container_width=True):
            st.session_state[position_key] -= 1
            st.rerun()
        return

    item = items[position]
    existing = store.get("audit", item["item_id"])

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
        stored_verdict = existing["verdict"] if existing else "approve"
        st.markdown("##### Verdict")
        verdict = st.radio(
            "Verdict",
            verdict_values,
            index=verdict_values.index(stored_verdict),
            horizontal=True,
            format_func=lambda value: value.replace("_", " ").title(),
            label_visibility="collapsed",
            key=f"verdict-{item['item_id']}",
        )

        with st.form(f"construction-{item['item_id']}"):
            if verdict == "needs_revision":
                comment = st.text_area(
                    "Comment",
                    value=existing["issues"] if existing else "",
                    placeholder="What should be revised?",
                )
            else:
                comment = ""
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
            store.save_audit(item["item_id"], verdict, comment.strip())
            st.session_state[position_key] += 1
            st.rerun()


if __name__ == "__main__":
    main()
