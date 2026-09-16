"""Administrator controls for accounts, assignments, progress, and export."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter, defaultdict

import streamlit as st

from annotation_app.assignment import (
    all_assignment_plan,
    quota_audit_assignment_plan,
    remaining_assignment_plan,
)
from benchmark_catalog import evaluation_groups
from annotation_app.storage import (
    AnnotationStore,
    generate_access_code,
)


EXPORT_BASE_FIELDS = ("annotation_id", "user_id", "item_id")
EXPORT_PAYLOAD_FIELDS = {
    "audit": ("verdict", "comment"),
    "eval": (
        "verdict",
        "primary_reason",
        "timing",
        "intensity",
        "comments",
        "diagnostics.detected_action_intensity",
        "diagnostics.action_intensity",
        "diagnostics.rejection_dimensions",
        "diagnostics.other_issue",
        "diagnostics.other_issue_detail",
    ),
}


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


def _flatten_payload(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        flattened = {}
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_payload(nested_value, nested_prefix))
        return flattened
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return {prefix: " | ".join(str(item) for item in value)}
        return {prefix: json.dumps(value, ensure_ascii=False, sort_keys=True)}
    return {prefix: "" if value is None else value}


def _export_csv(records: list[dict], review_type: str) -> bytes:
    buffer = io.StringIO()
    rows = []
    discovered_payload_fields = set()
    for record in records:
        payload_source = dict(record.get("payload", {}))
        if review_type == "eval":
            if "verdict" not in payload_source and "acceptable" in payload_source:
                payload_source["verdict"] = payload_source["acceptable"]
            payload_source.pop("acceptable", None)
        payload = _flatten_payload(payload_source)
        discovered_payload_fields.update(payload)
        rows.append(
            {
                **{key: record.get(key, "") for key in EXPORT_BASE_FIELDS},
                **payload,
            }
        )
    preferred_fields = EXPORT_PAYLOAD_FIELDS[review_type]
    extra_fields = sorted(discovered_payload_fields.difference(preferred_fields))
    fieldnames = (
        *EXPORT_BASE_FIELDS,
        *preferred_fields,
        *extra_fields,
    )
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
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
                _export_csv(matching_records, review_type),
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
            verdict = payload.get("verdict", payload.get("acceptable"))
            passing += verdict == "acceptable"
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
    task_label = st.radio(
        "Task type",
        ("Audit", "Eval"),
        horizontal=True,
    )
    review_type = task_label.lower()
    items = audit_items if review_type == "audit" else eval_items
    available_units = audit_units if review_type == "audit" else eval_units
    existing_type_ids = {
        row["user_id"]
        for row in store.progress_all_users()
        if row.get("review_type") == review_type
    }
    cohort_ids = (
        existing_type_ids
        | {
            user["user_id"]
            for user in store.list_users()
            if user["user_id"].startswith("audit")
        }
        if review_type == "audit"
        else existing_type_ids
    )
    if available_units == 0:
        st.caption(f"Existing accounts: {len(cohort_ids)}")
        st.info(f"No {task_label} items are available for assignment.")
        return
    if review_type == "audit" and len(cohort_ids) >= 10:
        st.info(
            f"Audit has {len(cohort_ids)} account records. New Audit account "
            "creation is disabled; use the restoration tool below to choose "
            "the 10 active accounts. Inactive records may remain so their "
            "saved annotations are preserved."
        )
        return

    with st.form("create-account-batch"):
        if review_type == "audit":
            assignment_scope = "Rebalance unfinished"
            st.caption(
                "Audit uses exactly 10 accounts and two reviews per item. "
                "Adding accounts redistributes pending tasks; submitted "
                "reviews stay with their authors."
            )
        else:
            assignment_scope = st.radio(
                "Scope",
                ("Rebalance unfinished", "All for new accounts"),
                horizontal=True,
            )
            if assignment_scope == "Rebalance unfinished":
                st.caption(
                    "Pending tasks are redistributed across existing and new "
                    "accounts. Existing task totals may change."
                )
            else:
                st.caption(
                    "Assign the full catalog to the new accounts in addition "
                    "to existing assignments."
                )
        st.caption(f"Existing accounts: {len(cohort_ids)}")
        max_new = (
            min(available_units, 10 - len(cohort_ids))
            if review_type == "audit"
            else available_units
        )
        count = st.number_input(
            "New annotators",
            min_value=1,
            max_value=max_new,
            value=max_new if review_type == "audit" else min(5, max_new),
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
            if assignment_scope == "Rebalance unfinished":
                assignment_users = [*sorted(cohort_ids), *user_ids]
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
                if assignment_scope == "Rebalance unfinished":
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


def _render_audit_reassignment(
    store: AnnotationStore,
    audit_items: list[dict],
) -> None:
    """Preview and restore an exact 10-account, 80-item Audit cohort."""
    progress = store.progress_all_users()
    audit_rows = {
        row["user_id"]: row
        for row in progress
        if row.get("review_type") == "audit" and row.get("assigned")
    }
    for user in store.list_users():
        if user["user_id"].startswith("audit"):
            audit_rows.setdefault(
                user["user_id"],
                {"user_id": user["user_id"], "assigned": 0, "submitted": 0},
            )
    if len(audit_rows) < 2 or not audit_items:
        return

    with st.expander("Restore 10 Audit accounts × 80 tasks"):
        st.caption(
            "Choose the 10 accounts that should each receive 80 of the 400 "
            "Audit items. Their submitted reviews stay fixed. Reviews already "
            "submitted by other accounts remain saved and exportable, while "
            "those accounts leave the active Audit assignment queue."
        )
        result = st.session_state.pop("audit_reassignment_done", None)
        if result:
            st.success(
                "Restored 800 active Audit assignments; "
                f"{result['preserved_annotations']} saved annotations were preserved."
            )
            if result["historical_submissions"]:
                st.caption(
                    f"{result['historical_submissions']} reviews from excluded "
                    "accounts remain in the annotation export."
                )
        original_ids = [f"audit{number:03d}" for number in range(1, 11)]
        defaults = (
            original_ids
            if set(original_ids).issubset(audit_rows)
            else sorted(audit_rows)
        )
        selected_ids = st.multiselect(
            "The 10 active Audit accounts",
            sorted(audit_rows),
            default=defaults,
            key="audit-reassignment-users",
        )
        if len(selected_ids) != 10 or len(audit_items) != 400:
            st.info("Select exactly 10 accounts from the 400-item Audit catalog.")
            return

        snapshot = store.assignment_snapshot("audit")
        submitted_users: dict[str, set[str]] = defaultdict(set)
        pending_users: dict[str, set[str]] = defaultdict(set)
        for row in snapshot:
            target = submitted_users if row["status"] == "submitted" else pending_users
            target[row["item_id"]].add(row["user_id"])
        try:
            plan = quota_audit_assignment_plan(
                selected_ids, audit_items, submitted_users, pending_users
            )
        except ValueError as error:
            st.error(str(error))
            return
        selected_set = set(selected_ids)
        saved_counts = store.annotation_counts("audit")
        preview = []
        for user_id, row in sorted(audit_rows.items()):
            completed = int(row.get("submitted") or 0)
            preview.append(
                {
                    "Account": user_id,
                    "Current tasks": int(row["assigned"]),
                    "After reassignment": completed + len(plan[user_id])
                    if user_id in selected_set
                    else 0,
                    "Saved annotations (unchanged)": saved_counts.get(user_id, 0),
                }
            )
        st.dataframe(preview, use_container_width=True, hide_index=True)
        current_by_item = Counter(row["item_id"] for row in snapshot)
        current_by_user = Counter(row["user_id"] for row in snapshot)
        catalog_ids = {item["item_id"] for item in audit_items}
        if (
            set(current_by_user) == selected_set
            and all(current_by_user[user_id] == 80 for user_id in selected_ids)
            and set(current_by_item) == catalog_ids
            and all(count == 2 for count in current_by_item.values())
        ):
            st.success("The selected 10 accounts already have 80 tasks each.")
            return
        if st.button(
            "Restore 10 × 80 Audit assignments",
            type="primary",
            use_container_width=True,
            key="apply-audit-reassignment",
        ):
            try:
                result = store.restore_audit_cohort(
                    selected_ids,
                    audit_items,
                    store.assignment_snapshot_hash(snapshot),
                )
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            else:
                st.session_state["audit_reassignment_done"] = result
                st.rerun()


def _render_audit_diagnostics(
    store: AnnotationStore,
    audit_items: list[dict],
) -> None:
    """Show a credential-free snapshot of Audit coverage and account workload."""
    assignments = store.assignment_snapshot("audit")
    annotation_counts = store.annotation_counts("audit")
    catalog_ids = {item["item_id"] for item in audit_items}
    coverage = Counter(
        row["item_id"] for row in assignments if row["item_id"] in catalog_ids
    )
    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in assignments:
        by_user[row["user_id"]].append(row)

    users = []
    for user in store.list_users():
        user_id = user["user_id"]
        rows = by_user.get(user_id, [])
        saved = annotation_counts.get(user_id, 0)
        if not rows and not saved and not user_id.startswith("audit"):
            continue
        users.append(
            {
                "Account": user_id,
                "Assigned": len(rows),
                "Completed": sum(row["status"] == "submitted" for row in rows),
                "Saved annotations": saved,
                "Created": user["created_at"],
                "Last assigned": max(
                    (row["assigned_at"] for row in rows), default=""
                ),
            }
        )

    report = {
        "catalog_items": len(catalog_ids),
        "audit_account_records": sum(
            user["Account"].startswith("audit") for user in users
        ),
        "active_audit_accounts": len(by_user),
        "assignment_rows": len(assignments),
        "submitted_assignment_rows": sum(
            row["status"] == "submitted" for row in assignments
        ),
        "saved_annotation_rows": sum(annotation_counts.values()),
        "catalog_coverage": {
            "zero_reviewers": sum(coverage[item_id] == 0 for item_id in catalog_ids),
            "one_reviewer": sum(coverage[item_id] == 1 for item_id in catalog_ids),
            "two_reviewers": sum(coverage[item_id] == 2 for item_id in catalog_ids),
            "more_than_two_reviewers": sum(
                coverage[item_id] > 2 for item_id in catalog_ids
            ),
            "assignments_outside_catalog": sum(
                row["item_id"] not in catalog_ids for row in assignments
            ),
        },
        "accounts": users,
    }
    with st.expander("Audit assignment diagnostics"):
        st.caption(
            "This report contains account IDs and counts only; it has no "
            "access codes or annotation text."
        )
        st.json({key: value for key, value in report.items() if key != "accounts"})
        st.dataframe(users, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Audit diagnostics",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name="proact-audit-diagnostics.json",
            mime="application/json",
            use_container_width=True,
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
        _render_audit_diagnostics(store, audit_items)
        _render_audit_reassignment(store, audit_items)
    with manage_tab:
        _render_manage(store)
    with progress_tab:
        records = store.export_annotations()
        _render_progress(store, records)
        _export_button(records)
