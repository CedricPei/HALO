"""Single-entry Streamlit application for cloud annotation."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import sys
import threading
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
CATALOG_VERSION = "human-eval-4000-2026-10-02-short-ids-v1"
INITIAL_ADMIN2_PASSWORD_HASH = (
    "pbkdf2_sha256$600000$e983fc7d21e20df94191e12090c5b63d$"
    "6a06af22d1cf83b423560c645e0b05606bdeb3b6de7f75eefff4f15fc2a0053f"
)
if str(ROOT_DIR) in sys.path:
    sys.path.remove(str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR))

import benchmark_catalog as catalog_module  # noqa: E402

# Cloud reruns can retain an imported module from before a function was renamed.
# Refresh it from the current checkout before any dependent imports bind names.
catalog_source_hash = hashlib.sha256(
    (ROOT_DIR / "benchmark_catalog.py").read_bytes()
).hexdigest()
with catalog_module.__dict__.setdefault("_annotation_reload_lock", threading.RLock()):
    if getattr(catalog_module, "_annotation_source_hash", None) != catalog_source_hash:
        importlib.invalidate_caches()
        catalog_module = importlib.reload(catalog_module)
        catalog_module._annotation_source_hash = catalog_source_hash
load_benchmark_items = catalog_module.load_benchmark_items

from annotation_app.eval_catalog import load_eval_release  # noqa: E402
from annotation_app.config import AppConfig  # noqa: E402
from annotation_app.review_views import (  # noqa: E402
    render_audit,
    render_eval,
)
import annotation_app.storage as storage_module  # noqa: E402
import annotation_app.assignment as assignment_module  # noqa: E402
from annotation_app.ui import apply_style  # noqa: E402


storage_source_hash = hashlib.sha256(
    (ROOT_DIR / "annotation_app" / "storage.py").read_bytes()
).hexdigest()
with storage_module.__dict__.setdefault("_annotation_reload_lock", threading.RLock()):
    if getattr(storage_module, "_annotation_source_hash", None) != storage_source_hash:
        importlib.invalidate_caches()
        storage_module = importlib.reload(storage_module)
        storage_module._annotation_source_hash = storage_source_hash
    AnnotationStore = storage_module.AnnotationStore
if not hasattr(assignment_module, "quota_audit_assignment_plan"):
    assignment_module = importlib.reload(assignment_module)
import annotation_app.admin_view as admin_view_module  # noqa: E402

if getattr(admin_view_module, "ADMIN_VIEW_VERSION", None) != (
    "2026-10-02-human-eval-cohort"
):
    admin_view_module = importlib.reload(admin_view_module)
render_admin = admin_view_module.render_admin


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
def _store(
    config: AppConfig,
    storage_cache_key: str,
    store_class_generation: int,
    _store_class: type[AnnotationStore],
) -> AnnotationStore:
    # A rolling deploy can cache an old class under a new on-disk source hash.
    # Key by the loaded class identity too, and construct that exact class.
    # Streamlit excludes the class argument itself (leading _) from hashing.
    return _store_class(config)


@st.cache_data(show_spinner="Loading Benchmark Audit items…")
def _audit_items(catalog_version: str) -> list[dict]:
    del catalog_version
    return load_benchmark_items()


@st.cache_data(show_spinner="Loading Agent Evaluation items…")
def _eval_release(catalog_version: str) -> tuple:
    del catalog_version
    return load_eval_release()


def _eval_items(catalog_version: str) -> list[dict]:
    return _eval_release(catalog_version)[0]


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
                "admin_source": "bootstrap",
            }
        else:
            user = store.authenticate_admin(user_id, access_code)
            if user is None:
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
                if key.startswith("cloud_") or key in {
                    "annotation_user",
                    "new_admin_credentials",
                    "new_assignment_credentials",
                }:
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
        store = _store(
            config, _storage_cache_key(), id(AnnotationStore), AnnotationStore
        )
    except Exception as error:
        st.title("Database connection failed")
        st.error("The App could not connect to its annotation database.")
        st.caption(
            "Check DATABASE_URL. For Supabase on Streamlit Cloud, use the pooled "
            f"connection string. Error type: {type(error).__name__}."
        )
        return

    try:
        # Historical resets and audit repairs have already been deployed.
        # New releases must not replay them against existing annotations.
        store.apply_eval_cohort(_eval_release(CATALOG_VERSION)[1])
        store.rename_user_accounts(
            "human-eval-4000-short-ids-v1",
            {f"eval4000-{number:02d}": f"eval{number:02d}" for number in range(1, 21)},
        )
    except (ValueError, RuntimeError) as error:
        st.title("Eval account setup paused")
        st.error(str(error))
        return

    try:
        store.apply_initial_admin(
            "create-admin2-2026-09-18",
            "admin2",
            INITIAL_ADMIN2_PASSWORD_HASH,
        )
    except ValueError as error:
        st.warning(f"Could not create admin2: {error}")

    assignment_version = CATALOG_VERSION
    if st.session_state.get("cloud_audit_assignment_version") != assignment_version:
        for key in list(st.session_state):
            if key.startswith(("cloud_audit_position_", "cloud_eval_position_")):
                del st.session_state[key]
        st.session_state["cloud_audit_assignment_version"] = assignment_version

    participant_users = {row["user_id"] for row in store.list_users()}
    user = st.session_state.get("annotation_user")
    if user and user.get("is_admin"):
        active_admin = (
            store.admin_exists(user["user_id"])
            if user.get("admin_source") == "database"
            else user["user_id"] == config.admin_username
        )
        if not active_admin:
            del st.session_state["annotation_user"]
            st.rerun()
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
