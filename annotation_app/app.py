"""Single-entry Streamlit application for cloud annotation."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
CATALOG_VERSION = "2026-09-10-complete-400-eval-purged"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from annotation_app.admin_view import render_admin  # noqa: E402
from benchmark_catalog import (  # noqa: E402
    load_construction_items,
    load_evaluation_items,
)
from annotation_app.config import AppConfig  # noqa: E402
from annotation_app.review_views import (  # noqa: E402
    render_audit,
    render_eval,
)
import annotation_app.storage as storage_module  # noqa: E402
from annotation_app.ui import apply_style  # noqa: E402


if not hasattr(
    storage_module.AnnotationStore,
    "apply_all_participant_accounts_purge",
):
    storage_module = importlib.reload(storage_module)
AnnotationStore = storage_module.AnnotationStore


st.set_page_config(
    page_title="Proactive Agent Benchmark Annotation Interface",
    page_icon="✓",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_style()
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding-bottom: .2rem;
      }
      [data-testid="stAppViewContainer"]:has(.copy-protection-marker)
        [data-testid="stMainBlockContainer"],
      [data-testid="stAppViewContainer"]:has(.copy-protection-marker)
        [data-testid="stSidebar"] {
        -webkit-user-select: none;
        user-select: none;
      }
      [data-testid="stAppViewContainer"]:has(.copy-protection-marker)
        input,
      [data-testid="stAppViewContainer"]:has(.copy-protection-marker)
        textarea {
        -webkit-user-select: text;
        user-select: text;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _storage_cache_key() -> str:
    source = Path(__file__).with_name("storage.py").read_bytes()
    return hashlib.sha256(source).hexdigest() + ":participant-purge-v1"


@st.cache_resource(show_spinner=False)
def _store(config: AppConfig, storage_cache_key: str) -> AnnotationStore:
    return AnnotationStore(config)


@st.cache_data(show_spinner="Loading Benchmark Audit items…")
def _audit_items(catalog_version: str) -> list[dict]:
    del catalog_version
    return load_construction_items()


@st.cache_data(show_spinner="Loading Agent Evaluation items…")
def _eval_items(catalog_version: str) -> list[dict]:
    del catalog_version
    return load_evaluation_items()


def _login(store: AnnotationStore, config: AppConfig) -> None:
    st.title("Proactive Agent Benchmark Annotation Interface")
    with st.form("annotation-login"):
        user_id = st.text_input("Participant ID")
        access_code = st.text_input("Access code", type="password")
        submitted = st.form_submit_button(
            "Sign in", type="primary", use_container_width=True
        )
    if submitted:
        if (
            user_id.strip().lower() == config.admin_username.lower()
            and hmac.compare_digest(access_code, config.admin_password)
        ):
            user = {
                "user_id": config.admin_username,
                "is_admin": True,
            }
        else:
            user = store.authenticate(user_id, access_code)
        if user:
            st.session_state["annotation_user"] = user
            st.rerun()
        st.error("Invalid participant ID or access code.")


def _sidebar_overview(task_label: str) -> None:
    task_overviews = {
        "Benchmark Audit": (
            "In Audit, assess whether the Scenario presents a plausible and "
            "well-supported proactive need, with enough evidence and a meaningful "
            "window in which assistance would be useful."
        ),
        "Agent Evaluation": (
            "In Eval, assess whether the Agent's action—or decision not to "
            "act—is justified by the information visible at this checkpoint, "
            "timely, relevant, safe, and proportionate."
        ),
    }
    task_overview = task_overviews.get(task_label)
    if not task_overview:
        return
    st.markdown("### Project overview")
    st.caption(
        "This project studies whether an AI assistant can recognize when "
        "proactive help would be useful, rather than only responding to an "
        "explicit request. Each item follows a person's situation as new "
        "information appears over time. The central question is whether the "
        "Agent should intervene now, and at what level of action."
    )
    st.caption(task_overview)
    st.markdown("**Trigger types**")
    st.caption(
        "- **Dependency:** A prerequisite for a future activity is missing, but "
        "no active attempt is blocked yet.\n"
        "- **Blockage:** An activity is already underway, but the current attempt "
        "has stopped and cannot continue without help.\n"
        "- **Risk:** No loss has happened yet, but inaction could plausibly lead "
        "to a future negative outcome.\n"
        "- **Opportunity:** Acting could create an optional benefit; doing nothing "
        "would cause no failure or loss."
    )


def _sidebar(user: dict, task_label: str) -> None:
    with st.sidebar:
        st.markdown("## Proactive Agent Benchmark Annotation Interface")
        st.caption(user["user_id"])
        st.markdown(f"**Task: {task_label}**")
        st.divider()
        _sidebar_overview(task_label)
        st.divider()
        st.caption("Supabase PostgreSQL")
        if st.button("Sign out", use_container_width=True):
            for key in list(st.session_state):
                if key.startswith("cloud_") or key == "annotation_user":
                    del st.session_state[key]
            st.rerun()


def main() -> None:
    config = AppConfig.load()
    missing = config.missing_secrets()
    if missing:
        st.title("Proactive Agent Benchmark Annotation Interface setup")
        st.error("Missing required secret(s): " + ", ".join(missing))
        st.markdown(
            "Add them to Streamlit Community Cloud Secrets. "
            "Do not commit secret values to GitHub."
        )
        return
    try:
        store = _store(config, _storage_cache_key())
        store.apply_catalog_reset(
            "complete-400-benchmark-2026-09-10-v2",
            _audit_items(CATALOG_VERSION),
            _eval_items(CATALOG_VERSION),
        )
        store.apply_review_type_purge(
            "purge-all-eval-data-2026-09-10",
            "eval",
        )
        store.apply_all_participant_accounts_purge(
            "purge-all-participant-accounts-2026-09-10",
        )
    except Exception as error:
        st.title("Database connection failed")
        st.error("The App could not connect to its annotation database.")
        st.caption(
            "Check DATABASE_URL. For Supabase on Streamlit Cloud, use the pooled "
            f"connection string. Error type: {type(error).__name__}."
        )
        return

    participant_users = {row["user_id"] for row in store.list_users()}
    user = st.session_state.get("annotation_user")
    if (
        user
        and not user.get("is_admin")
        and user["user_id"] not in participant_users
    ):
        del st.session_state["annotation_user"]
        st.rerun()
    if not user:
        _login(store, config)
        return

    if user.get("is_admin"):
        _sidebar(user, "Administration")
        render_admin(
            store,
            _audit_items(CATALOG_VERSION),
            _eval_items(CATALOG_VERSION),
        )
        return

    progress = store.progress_for_user(user["user_id"])
    review_type = next(
        (
            name
            for name in ("audit", "eval")
            if progress[name]["assigned"]
        ),
        None,
    )
    if review_type is None:
        _sidebar(user, "Awaiting assignment")
        st.title("No task assigned")
        st.info("Ask the administrator to assign an Audit or Eval task.")
        return
    if review_type == "audit":
        _sidebar(user, "Benchmark Audit")
        render_audit(store, user, _audit_items(CATALOG_VERSION))
    else:
        _sidebar(user, "Agent Evaluation")
        render_eval(store, user, _eval_items(CATALOG_VERSION))


if __name__ == "__main__":
    main()
