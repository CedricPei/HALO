"""Administrator controls for accounts, assignments, progress, and export."""

from __future__ import annotations

import csv
import io
import json

import streamlit as st

from annotation_app.assignment import apply_assignment_plan, balanced_assignment_plan
from annotation_app.storage import (
    AnnotationStore,
    generate_access_code,
    normalize_user_id,
)


def _credentials_csv(records: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=("user_id", "display_name", "role", "access_code"),
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
    st.dataframe(records, use_container_width=True, hide_index=True)
    st.download_button(
        "Download account codes",
        data=_credentials_csv(records),
        file_name="proact-anno-accounts.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )


def _create_legacy_accounts(store, packs: dict[str, dict]) -> list[dict]:
    existing_ids = {row["user_id"] for row in store.list_users()}
    created = []
    for user_id, assignments in sorted(packs.items()):
        if user_id in existing_ids:
            continue
        code = generate_access_code()
        store.create_user(user_id, user_id.title(), "annotator", code)
        store.assign_tasks(user_id, "audit", assignments["audit"])
        store.assign_tasks(user_id, "eval", assignments["eval"])
        created.append(
            {
                "user_id": user_id,
                "display_name": user_id.title(),
                "role": "annotator",
                "access_code": code,
            }
        )
    return created


def _render_accounts(
    store: AnnotationStore,
    audit_items: list[dict],
    eval_items: list[dict],
    packs: dict[str, dict],
) -> None:
    st.markdown("### Create accounts")
    st.caption(
        "Annotators sign in with a user ID and random access code; no email list is required."
    )
    _show_new_credentials()

    single_tab, batch_tab, legacy_tab = st.tabs(
        ["One account", "Balanced batch", "Current five packs"]
    )
    with single_tab:
        with st.form("create-single-account"):
            user_id = st.text_input("User ID", placeholder="annotator001")
            display_name = st.text_input("Display name", placeholder="Annotator 001")
            role = st.selectbox("Role", ("annotator", "reviewer", "admin"))
            pack_options = ["No assignment", *sorted(packs)]
            pack_name = st.selectbox("Optional assignment pack", pack_options)
            submitted = st.form_submit_button(
                "Create account", type="primary", use_container_width=True
            )
        if submitted:
            try:
                code = generate_access_code()
                account = store.create_user(user_id, display_name, role, code)
                if pack_name != "No assignment":
                    store.assign_tasks(
                        account["user_id"], "audit", packs[pack_name]["audit"]
                    )
                    store.assign_tasks(
                        account["user_id"], "eval", packs[pack_name]["eval"]
                    )
                record = {**account, "access_code": code}
                st.session_state["new_annotation_credentials"] = [record]
                store.log_admin_event(
                    "admin",
                    "create_user",
                    {"user_id": account["user_id"], "pack": pack_name},
                )
                st.rerun()
            except Exception as error:
                st.error(f"Could not create account: {error}")

    with batch_tab:
        with st.form("create-account-batch"):
            count = st.number_input(
                "Number of annotators", min_value=2, max_value=500, value=10
            )
            prefix = st.text_input("User ID prefix", value="annotator")
            audit_redundancy = st.number_input(
                "Audit reviews per Scenario",
                min_value=1,
                max_value=min(int(count), 5),
                value=min(2, int(count)),
            )
            st.caption(
                "Audit tasks are repeated across different people; all Event Prefixes "
                "from one Eval Scenario stay with the same person."
            )
            submitted = st.form_submit_button(
                "Create and assign batch", type="primary", use_container_width=True
            )
        if submitted:
            created = []
            try:
                existing_ids = {row["user_id"] for row in store.list_users()}
                user_ids = _next_ids(prefix, int(count), existing_ids)
                for index, new_user_id in enumerate(user_ids, start=1):
                    code = generate_access_code()
                    account = store.create_user(
                        new_user_id,
                        f"Annotator {index:03d}",
                        "annotator",
                        code,
                    )
                    created.append({**account, "access_code": code})
                plan = balanced_assignment_plan(
                    user_ids,
                    audit_items,
                    eval_items,
                    audit_redundancy=int(audit_redundancy),
                )
                assignment_count = apply_assignment_plan(store, plan)
                st.session_state["new_annotation_credentials"] = created
                store.log_admin_event(
                    "admin",
                    "create_balanced_batch",
                    {
                        "users": user_ids,
                        "audit_redundancy": int(audit_redundancy),
                        "assignments": assignment_count,
                    },
                )
                st.rerun()
            except Exception as error:
                st.error(
                    f"Batch creation stopped after {len(created)} account(s): {error}"
                )

    with legacy_tab:
        st.write(
            "Create `reviewer1` through `reviewer5` with the exact assignments from "
            "the existing local pilot. Existing accounts are left unchanged."
        )
        if st.button(
            "Create current pilot accounts",
            type="primary",
            use_container_width=True,
            disabled=not packs,
        ):
            try:
                created = _create_legacy_accounts(store, packs)
                st.session_state["new_annotation_credentials"] = created
                store.log_admin_event(
                    "admin",
                    "create_legacy_accounts",
                    {"users": [row["user_id"] for row in created]},
                )
                st.rerun()
            except Exception as error:
                st.error(f"Could not create the pilot accounts: {error}")

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
    packs: dict[str, dict],
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
        _render_accounts(store, audit_items, eval_items, packs)
    with progress_tab:
        _render_progress(store)
    with export_tab:
        _render_export(store)
