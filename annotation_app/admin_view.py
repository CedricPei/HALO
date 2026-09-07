"""Administrator controls for accounts, assignments, progress, and export."""

from __future__ import annotations

import csv
import io
import json
import zipfile

import streamlit as st

from annotation_app.assignment import (
    all_assignment_plan,
    remaining_assignment_plan,
)
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


def _export_archive(records: list[dict]) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for review_type in ("audit", "eval"):
            matching_records = [
                record for record in records if record["review_type"] == review_type
            ]
            bundle.writestr(
                f"proact-anno-{review_type}-annotations.csv",
                _export_csv(matching_records),
            )
    return archive.getvalue()


def _pass_rate(records: list[dict]) -> float | None:
    if not records:
        return None
    passing = 0
    for record in records:
        payload = record.get("payload", {})
        if record.get("review_type") == "audit":
            passing += payload.get("verdict") == "approve"
        elif record.get("review_type") == "eval":
            passing += payload.get("acceptable") == "acceptable"
    return passing / len(records)


def _next_ids(prefix: str, count: int, existing_ids: set[str]) -> list[str]:
    result = []
    number = 1
    while len(result) < count:
        candidate = f"{prefix}{number:03d}"
        number += 1
        if candidate not in existing_ids:
            result.append(candidate)
    return result


def _show_credentials(state_key: str, message: str, button_key: str) -> None:
    records = st.session_state.get(state_key, [])
    if not records:
        return
    st.success(message)
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
        key=button_key,
    )


def _render_assignments(
    store: AnnotationStore,
    audit_items: list[dict],
    eval_items: list[dict],
) -> None:
    _show_credentials(
        "new_assignment_credentials",
        "Assignments created. Download the access codes now.",
        "download-assignment-codes",
    )

    audit_units = len(audit_items)
    eval_units = len(evaluation_groups(eval_items))
    max_batch_size = max(1, min(audit_units, eval_units))
    task_label = st.radio(
        "Task type",
        ("Audit", "Eval"),
        horizontal=True,
    )
    review_type = task_label.lower()
    existing_type_ids = {
        row["user_id"]
        for row in store.progress_all_users()
        if row.get("review_type") == review_type
    }
    with st.form("create-account-batch"):
        assignment_scope = st.radio(
            "Scope",
            ("Remaining", "All"),
            horizontal=True,
        )
        st.caption(f"Existing accounts: {len(existing_type_ids)}")
        count = st.number_input(
            "New annotators",
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
        try:
            users = store.list_users()
            existing_ids = {row["user_id"] for row in users}
            user_ids = _next_ids(review_type, int(count), existing_ids)
            items = audit_items if review_type == "audit" else eval_items
            if assignment_scope == "Remaining":
                assignment_users = [*sorted(existing_type_ids), *user_ids]
                plan = remaining_assignment_plan(
                    assignment_users,
                    review_type,
                    items,
                    store.submitted_users_by_item(review_type),
                )
            else:
                plan = all_assignment_plan(user_ids, review_type, items)

            if not any(plan.values()):
                st.info(f"No remaining {task_label} tasks to assign.")
            else:
                for new_user_id in user_ids:
                    code = generate_access_code()
                    store.create_user(new_user_id, code)
                    created.append({"user_id": new_user_id, "access_code": code})
                if assignment_scope == "Remaining":
                    store.replace_pending_assignments(review_type, plan)
                else:
                    for user_id, item_ids in plan.items():
                        store.assign_tasks(user_id, review_type, item_ids)
                st.session_state["new_assignment_credentials"] = created
                st.rerun()
        except Exception as error:
            st.error(
                f"Batch creation stopped after {len(created)} account(s): {error}"
            )


def _render_manage(store: AnnotationStore) -> None:
    deleted_ids = st.session_state.pop("deleted_account_ids", [])
    if deleted_ids:
        st.success(f"Deleted {len(deleted_ids)} account(s).")

    accounts = store.list_user_codes()
    if not accounts:
        st.info("No annotator accounts have been created.")
        return
    st.download_button(
        "Download codes",
        data=_credentials_csv(accounts),
        file_name="proact-anno-accounts.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    account_ids = [row["user_id"] for row in accounts]
    delete_ids = st.multiselect(
        "Delete accounts",
        account_ids,
        placeholder="Select accounts",
        key="manage-delete-selection",
        on_change=lambda: st.session_state.update(
            {"confirm-delete-accounts": False}
        ),
    )
    confirmed = st.checkbox(
        "Also delete their assignments and annotations",
        disabled=not delete_ids,
        key="confirm-delete-accounts",
    )
    if st.button(
        "Delete selected accounts",
        disabled=not delete_ids or not confirmed,
        use_container_width=True,
        key="delete-accounts",
    ):
        store.delete_users(delete_ids)
        deleted_set = set(delete_ids)
        st.session_state["new_assignment_credentials"] = [
            row
            for row in st.session_state.get("new_assignment_credentials", [])
            if row["user_id"] not in deleted_set
        ]
        for state_key in (
            "manage-delete-selection",
            "confirm-delete-accounts",
        ):
            st.session_state.pop(state_key, None)
        st.session_state["deleted_account_ids"] = delete_ids
        st.rerun()


def _progress_metrics(
    assigned: int,
    submitted: int,
    pass_rate: float | None,
) -> None:
    columns = st.columns(5)
    columns[0].metric("Assigned", assigned)
    columns[1].metric("Completed", submitted)
    columns[2].metric("Remaining", max(assigned - submitted, 0))
    columns[3].metric(
        "Completion",
        f"{(100 * submitted / assigned):.1f}%" if assigned else "—",
    )
    columns[4].metric(
        "Pass rate",
        f"{(100 * pass_rate):.1f}%" if pass_rate is not None else "—",
        help="Approve for Audit; Acceptable for Eval.",
    )


def _render_progress(store: AnnotationStore, records: list[dict]) -> None:
    rows = store.progress_all_users()
    if not rows:
        st.info("No accounts or assignments yet.")
        return

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
        review_records = [
            record for record in records if record["review_type"] == review_type
        ]
        st.markdown(f"#### {label}")
        _progress_metrics(assigned, submitted, _pass_rate(review_records))

    st.markdown("#### Annotator details")
    users = store.list_users()
    users_by_id = {row["user_id"]: row for row in users}
    selected_id = st.selectbox(
        "Annotator",
        [row["user_id"] for row in users],
        index=None,
        placeholder="Select an annotator",
        key="progress-annotator-selection",
    )
    if selected_id is None:
        return
    selected = users_by_id[selected_id]
    selected_rows = [row for row in rows if row["user_id"] == selected_id]
    assigned = sum(int(row.get("assigned") or 0) for row in selected_rows)
    submitted = sum(int(row.get("submitted") or 0) for row in selected_rows)
    task_types = ", ".join(
        sorted(
            {
                str(row["review_type"]).title()
                for row in selected_rows
                if row.get("review_type")
            }
        )
    ) or "Unassigned"
    last_login = selected.get("last_login_at") or "Never"
    st.caption(f"Task: {task_types} · Last login: {last_login}")
    selected_records = [
        record for record in records if record["user_id"] == selected_id
    ]
    _progress_metrics(assigned, submitted, _pass_rate(selected_records))


def _export_button(records: list[dict]) -> None:
    st.download_button(
        "Download annotations",
        data=_export_archive(records),
        file_name="proact-anno-annotations.zip",
        mime="application/zip",
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
    assignment_tab, manage_tab, progress_tab = st.tabs(
        ["Assignments", "Manage", "Progress"]
    )
    with assignment_tab:
        _render_assignments(store, audit_items, eval_items)
    with manage_tab:
        _render_manage(store)
    with progress_tab:
        records = store.export_annotations()
        _render_progress(store, records)
        _export_button(records)
