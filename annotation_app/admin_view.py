"""Administrator controls for accounts, assignments, progress, and export."""

from __future__ import annotations

import csv
import io
import json

import streamlit as st

from annotation_app.assignment import apply_assignment_plan, balanced_assignment_plan
from annotation_app.catalog import evaluation_groups
from annotation_app.storage import (
    AnnotationStore,
    generate_access_code,
    normalize_user_id,
)


def _credentials_csv(records: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=("user_id", "access_code"),
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8-sig")


def _export_csv(records: list[dict]) -> bytes:
    buffer = io.StringIO()
    fieldnames = (
        "revision_id",
        "user_id",
        "review_type",
        "item_id",
        "state",
        "created_at",
        "payload_json",
    )
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                **{key: record.get(key, "") for key in fieldnames},
                "payload_json": json.dumps(
                    record["payload"], ensure_ascii=False, sort_keys=True
                ),
            }
        )
    return buffer.getvalue().encode("utf-8-sig")


def _next_ids(prefix: str, count: int, existing_ids: set[str]) -> list[str]:
    prefix = normalize_user_id(prefix)
    result = []
    number = 1
    while len(result) < count:
        candidate = f"{prefix}{number:03d}"
        number += 1
        if candidate not in existing_ids:
            result.append(candidate)
    return result


def _show_new_credentials() -> None:
    records = st.session_state.get("new_annotation_credentials", [])
    if not records:
        return
    st.success(
        "Accounts created. Download the codes now; only their secure hashes are stored."
    )
    st.dataframe(
        [
            {"user_id": row["user_id"], "access_code": row["access_code"]}
            for row in records
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download account codes",
        data=_credentials_csv(records),
        file_name="proact-anno-accounts.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )


def _render_accounts(
    store: AnnotationStore,
    audit_items: list[dict],
    eval_items: list[dict],
) -> None:
    st.markdown("### Create accounts")
    st.caption(
        "Choose one task type and the number of annotators. Accounts, access codes, "
        "and balanced assignments are generated automatically."
    )
    _show_new_credentials()

    audit_units = len(audit_items)
    eval_units = len(evaluation_groups(eval_items))
    max_batch_size = max(1, min(audit_units, eval_units))
    with st.form("create-account-batch"):
        task_label = st.radio(
            "Task type",
            ("Audit", "Eval"),
            horizontal=True,
        )
        count = st.number_input(
            "Number of annotators",
            min_value=1,
            max_value=max_batch_size,
            value=min(5, max_batch_size),
            step=1,
        )
        st.caption(
            "Audit Scenarios go to two annotators when the batch has at least two "
            "people. Eval checkpoints are assigned once, and checkpoints from the "
            "same Scenario always stay with one annotator."
        )
        submitted = st.form_submit_button(
            "Create accounts and assign tasks",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        created = []
        review_type = task_label.lower()
        try:
            existing_ids = {row["user_id"] for row in store.list_users()}
            user_ids = _next_ids(review_type, int(count), existing_ids)
            audit_redundancy = min(2, len(user_ids)) if review_type == "audit" else 1
            plan = balanced_assignment_plan(
                user_ids,
                audit_items if review_type == "audit" else [],
                eval_items if review_type == "eval" else [],
                audit_redundancy=audit_redundancy,
            )
            for new_user_id in user_ids:
                code = generate_access_code()
                suffix = new_user_id.removeprefix(review_type)
                account = store.create_user(
                    new_user_id,
                    f"{task_label} Annotator {suffix}",
                    "annotator",
                    code,
                )
                created.append({**account, "access_code": code})
            assignment_count = apply_assignment_plan(store, plan)
            st.session_state["new_annotation_credentials"] = created
            store.log_admin_event(
                "admin",
                "create_single_type_batch",
                {
                    "review_type": review_type,
                    "users": user_ids,
                    "audit_redundancy": audit_redundancy,
                    "assignments": assignment_count,
                },
            )
            st.rerun()
        except Exception as error:
            st.error(
                f"Batch creation stopped after {len(created)} account(s): {error}"
            )

    st.markdown("### Manage accounts")
    users = store.list_users()
    if not users:
        st.info("No annotator accounts have been created.")
        return
    st.dataframe(users, use_container_width=True, hide_index=True)
    selected_id = st.selectbox(
        "Account",
        [row["user_id"] for row in users],
        key="manage-account-id",
    )
    selected = next(row for row in users if row["user_id"] == selected_id)
    action_col, reset_col = st.columns(2)
    desired_active = not bool(int(selected["active"]))
    if action_col.button(
        "Enable account" if desired_active else "Disable account",
        use_container_width=True,
    ):
        store.set_user_active(selected_id, desired_active)
        store.log_admin_event(
            "admin",
            "set_user_active",
            {"user_id": selected_id, "active": desired_active},
        )
        st.rerun()
    if reset_col.button("Generate new access code", use_container_width=True):
        code = generate_access_code()
        store.reset_access_code(selected_id, code)
        st.session_state["new_annotation_credentials"] = [
            {
                "user_id": selected_id,
                "display_name": selected["display_name"],
                "role": selected["role"],
                "access_code": code,
            }
        ]
        store.log_admin_event(
            "admin", "reset_access_code", {"user_id": selected_id}
        )
        st.rerun()


def _render_progress(store: AnnotationStore) -> None:
    st.markdown("### Progress")
    rows = store.progress_all_users()
    if not rows:
        st.info("No accounts or assignments yet.")
        return
    display_rows = []
    for row in rows:
        assigned = int(row.get("assigned") or 0)
        submitted = int(row.get("submitted") or 0)
        display_rows.append(
            {
                "user_id": row["user_id"],
                "display_name": row["display_name"],
                "role": row["role"],
                "active": bool(int(row["active"])),
                "review_type": row.get("review_type") or "—",
                "submitted": submitted,
                "assigned": assigned,
                "completion": f"{(100 * submitted / assigned):.1f}%" if assigned else "—",
            }
        )
    st.dataframe(display_rows, use_container_width=True, hide_index=True)


def _render_export(store: AnnotationStore) -> None:
    st.markdown("### Export")
    records = store.export_latest()
    st.metric("Latest submitted/draft records", len(records))
    st.caption(
        f"The database contains {store.revision_count()} immutable revision(s). "
        "This export selects only the latest revision for each user and item."
    )
    st.download_button(
        "Download latest annotations",
        data=_export_csv(records),
        file_name="proact-anno-latest.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
        disabled=not records,
    )


def render_admin(
    store: AnnotationStore,
    config,
    audit_items: list[dict],
    eval_items: list[dict],
) -> None:
    st.title("Administration")
    st.caption(
        f"Backend: {config.backend_name} · {len(audit_items)} Audit items · "
        f"{len(eval_items)} annotatable Eval checkpoints"
    )
    account_tab, progress_tab, export_tab = st.tabs(
        ["Accounts & assignments", "Progress", "Export"]
    )
    with account_tab:
        _render_accounts(store, audit_items, eval_items)
    with progress_tab:
        _render_progress(store)
    with export_tab:
        _render_export(store)
