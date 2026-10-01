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
    generate_admin_password,
)

ADMIN_VIEW_VERSION = "2026-10-02-human-eval-cohort"


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


def _render_eval_assignments(
    store: AnnotationStore,
    eval_items: list[dict],
) -> None:
    _show_credentials(
        "new_assignment_credentials",
        "Assignments created. Download the access codes now.",
        "download-assignment-codes",
    )

    eval_units = len(evaluation_groups(eval_items))
    existing_ids = {
        row["user_id"]
        for row in store.progress_all_users()
        if row.get("review_type") == "eval"
    }
    if eval_units == 0:
        st.info("No Eval items are available for assignment.")
        return

    with st.form("create-eval-account-batch"):
        assignment_scope = st.radio(
            "Scope",
            ("Rebalance unfinished", "All for new accounts"),
            horizontal=True,
        )
        if assignment_scope == "Rebalance unfinished":
            st.caption(
                "Pending Eval tasks are redistributed across existing and new "
                "accounts. Existing task totals may change."
            )
        else:
            st.caption(
                "Assign the full Eval catalog to the new accounts in addition "
                "to existing assignments."
            )
        st.caption(f"Existing Eval accounts: {len(existing_ids)}")
        count = st.number_input(
            "New Eval annotators",
            min_value=1,
            max_value=eval_units,
            value=min(5, eval_units),
            step=1,
        )
        submitted = st.form_submit_button(
            "Create Eval assignments",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        created = []
        try:
            account_ids = {row["user_id"] for row in store.list_users()}
            user_ids = _next_ids("eval", int(count), account_ids)
            if assignment_scope == "Rebalance unfinished":
                plan = remaining_assignment_plan(
                    [*sorted(existing_ids), *user_ids],
                    "eval",
                    eval_items,
                    store.submitted_users_by_item("eval"),
                )
            else:
                plan = all_assignment_plan(user_ids, "eval", eval_items)

            if not any(plan.values()):
                st.info("No remaining Eval tasks to assign.")
            else:
                for user_id in user_ids:
                    code = generate_access_code()
                    store.create_user(user_id, code)
                    created.append({"user_id": user_id, "access_code": code})
                if assignment_scope == "Rebalance unfinished":
                    store.replace_pending_assignments("eval", plan)
                else:
                    for user_id, item_ids in plan.items():
                        store.assign_tasks(user_id, "eval", item_ids)
                st.session_state["new_assignment_credentials"] = created
                st.rerun()
        except Exception as error:
            st.error(
                f"Batch creation stopped after {len(created)} account(s): {error}"
            )


def _render_admin_accounts(store: AnnotationStore) -> None:
    st.markdown("#### Administrator accounts")
    created = st.session_state.get("new_admin_credentials")
    if created:
        st.success(
            f"Administrator {created['user_id']} created. Save this password now."
        )
        st.code(created["password"], language=None)
        st.caption("This password is shown only in this browser session.")
        if st.button("I saved the password", key="dismiss-admin-password"):
            st.session_state.pop("new_admin_credentials", None)
            st.rerun()

    admins = store.list_admins()
    st.dataframe(
        [
            {
                "Administrator": store.config.admin_username,
                "Source": "Streamlit Secrets",
                "Last login": "—",
            },
            *[
                {
                    "Administrator": row["user_id"],
                    "Source": "Database",
                    "Last login": row["last_login_at"] or "Never",
                }
                for row in admins
            ],
        ],
        use_container_width=True,
        hide_index=True,
    )
    with st.form("create-administrator"):
        existing_ids = {row["user_id"] for row in admins}
        next_number = 2
        while f"admin{next_number}" in existing_ids:
            next_number += 1
        user_id = st.text_input(
            "New administrator ID", value=f"admin{next_number}"
        )
        submitted = st.form_submit_button("Create administrator")
    if submitted:
        password = generate_admin_password()
        try:
            store.create_admin(user_id, password)
        except Exception as error:
            st.error(f"Could not create administrator: {error}")
        else:
            st.session_state["new_admin_credentials"] = {
                "user_id": user_id.strip().lower().replace(" ", "-"),
                "password": password,
            }
            st.rerun()

    if admins:
        with st.expander("Revoke a database administrator"):
            with st.form("revoke-administrator", clear_on_submit=True):
                revoke_id = st.selectbox(
                    "Administrator to revoke",
                    [row["user_id"] for row in admins],
                )
                confirmed = st.checkbox(
                    "I understand this account will lose access"
                )
                revoke = st.form_submit_button("Revoke administrator")
            if revoke and confirmed:
                store.delete_admin(revoke_id)
                st.session_state.pop("new_admin_credentials", None)
                st.rerun()


def _render_manage(store: AnnotationStore) -> None:
    _render_admin_accounts(store)
    st.divider()
    st.markdown("#### Annotator accounts")
    deleted_ids = st.session_state.pop("deleted_account_ids", [])
    if deleted_ids:
        st.success(f"Deleted {len(deleted_ids)} account(s).")

    accounts = store.list_user_codes()
    if not accounts:
        st.info("No annotator accounts have been created.")
        return
    if any(not account["access_code"] for account in accounts):
        st.caption("Pre-provisioned Eval codes are stored as hashes. Use the original "
                   "credential handout; their codes are blank in this export.")
    st.download_button(
        "Download codes",
        data=_credentials_csv(accounts),
        file_name="proact-anno-accounts.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    fixed_audit_ids = {f"audit{number:03d}" for number in range(1, 11)}
    account_ids = [
        row["user_id"]
        for row in accounts
        if row["user_id"] not in fixed_audit_ids
    ]
    st.caption("The fixed audit001–audit010 accounts cannot be deleted here.")
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
    task_label = st.radio(
        "Task type",
        ("Audit", "Eval"),
        horizontal=True,
        key="progress-review-type",
    )
    review_type = task_label.lower()
    users_by_id = {user["user_id"]: user for user in store.list_users()}
    records_by_user: dict[str, list[dict]] = {}
    for record in records:
        if record["review_type"] == review_type:
            records_by_user.setdefault(record["user_id"], []).append(record)

    details = []
    for row in rows:
        if row.get("review_type") != review_type:
            continue
        user_id = row["user_id"]
        assigned = int(row.get("assigned") or 0)
        submitted = int(row.get("submitted") or 0)
        pass_rate = _pass_rate(records_by_user.get(user_id, []))
        details.append(
            {
                "Annotator": user_id,
                "Assigned": assigned,
                "Completed": submitted,
                "Remaining": max(assigned - submitted, 0),
                "Completion": (
                    f"{100 * submitted / assigned:.1f}%" if assigned else "—"
                ),
                "Pass rate": f"{100 * pass_rate:.1f}%" if pass_rate is not None else "—",
                "Last login": (
                    users_by_id.get(user_id, {}).get("last_login_at") or "Never"
                ),
            }
        )
    if details:
        st.dataframe(details, use_container_width=True, hide_index=True)
    else:
        st.info(f"No {task_label} assignments yet.")


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
        ["Eval Assignments", "Manage", "Progress"]
    )
    with assignment_tab:
        _render_eval_assignments(store, eval_items)
    with manage_tab:
        _render_manage(store)
    with progress_tab:
        records = store.export_annotations()
        _render_progress(store, records)
        _export_button(records)
