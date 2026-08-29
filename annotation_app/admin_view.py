"""Administrator controls for accounts, assignments, progress, and export."""

from __future__ import annotations

import csv
import io
import json

import streamlit as st

from annotation_app.assignment import assignment_plan
from benchmark_catalog import evaluation_groups
from annotation_app.storage import (
    AnnotationStore,
    generate_access_code,
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
        "annotation_id",
        "user_id",
        "review_type",
        "item_id",
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
        submitted = st.form_submit_button(
            "Create assignments",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        created = []
        review_type = task_label.lower()
        try:
            existing_ids = {row["user_id"] for row in store.list_users()}
            user_ids = _next_ids(review_type, int(count), existing_ids)
            plan = assignment_plan(
                user_ids,
                review_type,
                audit_items if review_type == "audit" else eval_items,
            )
            for new_user_id in user_ids:
                code = generate_access_code()
                suffix = new_user_id.removeprefix(review_type)
                account = store.create_user(
                    new_user_id,
                    f"{task_label} Annotator {suffix}",
                    code,
                )
                created.append({**account, "access_code": code})
            for user_id, item_ids in plan.items():
                store.assign_tasks(user_id, review_type, item_ids)
            st.session_state["new_annotation_credentials"] = created
            st.rerun()
        except Exception as error:
            st.error(
                f"Batch creation stopped after {len(created)} account(s): {error}"
            )

    st.markdown("#### Manage accounts")
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
        st.rerun()


def _render_progress(store: AnnotationStore) -> None:
    rows = store.progress_all_users()
    if not rows:
        st.info("No accounts or assignments yet.")
        return

    total_assigned = sum(int(row.get("assigned") or 0) for row in rows)
    total_submitted = sum(int(row.get("submitted") or 0) for row in rows)
    total_remaining = max(total_assigned - total_submitted, 0)
    total_completion = (
        f"{(100 * total_submitted / total_assigned):.1f}%"
        if total_assigned
        else "—"
    )
    overall_columns = st.columns(4)
    overall_columns[0].metric("Total assigned reviews", total_assigned)
    overall_columns[1].metric("Completed", total_submitted)
    overall_columns[2].metric("Remaining", total_remaining)
    overall_columns[3].metric("Overall completion", total_completion)

    st.markdown("#### By task type")
    type_rows = []
    for review_type, label in (("audit", "Audit"), ("eval", "Eval")):
        assigned = sum(
            int(row.get("assigned") or 0)
            for row in rows
            if row.get("review_type") == review_type
        )
        submitted = sum(
            int(row.get("submitted") or 0)
            for row in rows
            if row.get("review_type") == review_type
        )
        type_rows.append(
            {
                "task_type": label,
                "assigned": assigned,
                "completed": submitted,
                "remaining": max(assigned - submitted, 0),
                "completion": (
                    f"{(100 * submitted / assigned):.1f}%" if assigned else "—"
                ),
            }
        )
    st.dataframe(type_rows, use_container_width=True, hide_index=True)
    st.caption(
        "Totals count assigned reviews. An Audit Scenario assigned for independent "
        "double review contributes two assigned reviews."
    )

    st.markdown("#### Annotator details")
    users = store.list_users()
    users_by_id = {row["user_id"]: row for row in users}
    selected_id = st.selectbox(
        "Annotator",
        [row["user_id"] for row in users],
        format_func=lambda user_id: (
            f"{user_id} · {users_by_id[user_id]['display_name']}"
        ),
        key="progress-annotator-id",
    )
    selected = users_by_id[selected_id]
    selected_rows = [row for row in rows if row["user_id"] == selected_id]
    assigned = sum(int(row.get("assigned") or 0) for row in selected_rows)
    submitted = sum(int(row.get("submitted") or 0) for row in selected_rows)
    remaining = max(assigned - submitted, 0)
    completion = f"{(100 * submitted / assigned):.1f}%" if assigned else "—"
    task_types = ", ".join(
        sorted(
            {
                str(row["review_type"]).title()
                for row in selected_rows
                if row.get("review_type")
            }
        )
    ) or "Unassigned"
    status = "Active" if bool(int(selected["active"])) else "Disabled"
    last_login = selected.get("last_login_at") or "Never"
    st.caption(
        f"{status} · Task: {task_types} · Last login: {last_login}"
    )
    annotator_columns = st.columns(4)
    annotator_columns[0].metric("Assigned", assigned)
    annotator_columns[1].metric("Completed", submitted)
    annotator_columns[2].metric("Remaining", remaining)
    annotator_columns[3].metric("Completion", completion)


def _export_button(store: AnnotationStore) -> None:
    records = store.export_annotations()
    st.download_button(
        "Export annotations",
        data=_export_csv(records),
        file_name="proact-anno-annotations.csv",
        mime="text/csv",
        type="primary",
        disabled=not records,
    )


def render_admin(
    store: AnnotationStore,
    audit_items: list[dict],
    eval_items: list[dict],
) -> None:
    st.title("Administration")
    st.caption(
        f"Backend: Supabase PostgreSQL · {len(audit_items)} Audit items · "
        f"{len(eval_items)} annotatable Eval checkpoints"
    )
    account_tab, progress_tab = st.tabs(["Assignments", "Progress"])
    with account_tab:
        _render_accounts(store, audit_items, eval_items)
    with progress_tab:
        _render_progress(store)
        _export_button(store)
