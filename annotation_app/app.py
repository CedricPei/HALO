"""Single-entry Streamlit application for cloud annotation."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import sys
import threading
from pathlib import Path
from types import ModuleType

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
CATALOG_VERSION = "human-eval-4000-2026-10-02-short-ids-v1"
if str(ROOT_DIR) in sys.path:
    sys.path.remove(str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR))


def _refresh_module(module: ModuleType, *dependencies: str) -> tuple[ModuleType, str]:
    """Reload changed code before dependent views bind its functions."""
    paths = [ROOT_DIR / path for path in dependencies] + [Path(module.__file__)]
    source_hash = hashlib.sha256(b"\0".join(path.read_bytes() for path in paths)).hexdigest()
    with module.__dict__.setdefault("_annotation_reload_lock", threading.RLock()):
        if getattr(module, "_annotation_source_hash", None) != source_hash:
            importlib.invalidate_caches()
            module = importlib.reload(module)
            module._annotation_source_hash = source_hash
    return module, source_hash


import benchmark_catalog as catalog_module  # noqa: E402

catalog_module, catalog_source_hash = _refresh_module(catalog_module)
load_benchmark_items = catalog_module.load_benchmark_items
audit_release_key = (catalog_source_hash, id(load_benchmark_items))

from annotation_app.config import AppConfig  # noqa: E402

# Refresh dependencies first; file signatures keep widget reruns inexpensive.
import evaluation.prompt as agent_prompt_module  # noqa: E402

agent_prompt_module, _ = _refresh_module(
    agent_prompt_module, "env/app_surfaces/evaluated_agent.md"
)
import evaluation.intensity as intensity_module  # noqa: E402

intensity_module, _ = _refresh_module(
    intensity_module, "env/app_surfaces/evaluated_agent.md", "evaluation/prompt.py"
)
import annotation_app.storage as storage_module  # noqa: E402

storage_module, storage_source_hash = _refresh_module(
    storage_module, "annotation_app/schema.sql"
)
import annotation_app.assignment as assignment_module  # noqa: E402

assignment_module, _ = _refresh_module(assignment_module, "benchmark_catalog.py")
import annotation_app.ui as ui_module  # noqa: E402

ui_module, _ = _refresh_module(ui_module)
import annotation_app.eval_catalog as eval_catalog_module  # noqa: E402

eval_catalog_module, eval_source_hash = _refresh_module(eval_catalog_module)
import annotation_app.review_views as review_views_module  # noqa: E402

review_views_module, _ = _refresh_module(
    review_views_module, "env/app_surfaces/evaluated_agent.md", "evaluation/prompt.py",
    "evaluation/intensity.py", "annotation_app/storage.py", "annotation_app/ui.py"
)
import annotation_app.admin_view as admin_view_module  # noqa: E402

admin_view_module, _ = _refresh_module(
    admin_view_module, "annotation_app/storage.py", "annotation_app/assignment.py",
    "benchmark_catalog.py"
)

AnnotationStore = storage_module.AnnotationStore
load_eval_release = eval_catalog_module.load_eval_release
eval_release_key = (eval_source_hash, id(load_eval_release),
                    eval_catalog_module.release_signature())
apply_style = ui_module.apply_style
render_audit = review_views_module.render_audit
render_eval = review_views_module.render_eval
render_admin = admin_view_module.render_admin


st.set_page_config(
    page_title="HALO Annotation",
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


@st.cache_resource(show_spinner="Loading Benchmark Audit items…")
def _audit_items(catalog_version: str, source_key: tuple) -> list[dict]:
    # Published source data is shared read-only across all review sessions.
    del catalog_version, source_key
    return load_benchmark_items()


@st.cache_resource(show_spinner="Loading Agent Evaluation items…")
def _eval_items(catalog_version: str, release_key: tuple) -> list[dict]:
    # Avoid copying the entire frozen release on every widget interaction.
    # Callers must not mutate the shared items.
    del catalog_version, release_key
    return load_eval_release()


def _login(store: AnnotationStore, config: AppConfig) -> None:
    st.title("HALO Annotation")
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
        st.markdown("## HALO Annotation")
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
        st.title("HALO Annotation setup")
        st.error("Missing required secret(s): " + ", ".join(missing))
        st.markdown(
            "Add them to Streamlit Community Cloud Secrets. "
            "Do not commit secret values to GitHub."
        )
        return
    try:
        store = _store(
            config, storage_source_hash, id(AnnotationStore), AnnotationStore
        )
    except Exception as error:
        st.title("Database connection failed")
        st.error("The App could not connect to its annotation database.")
        st.caption(
            "Check DATABASE_URL. For Supabase on Streamlit Cloud, use the pooled "
            f"connection string. Error type: {type(error).__name__}."
        )
        return

    assignment_version = CATALOG_VERSION
    if st.session_state.get("cloud_audit_assignment_version") != assignment_version:
        for key in list(st.session_state):
            if key.startswith(("cloud_audit_position_", "cloud_eval_position_")):
                del st.session_state[key]
        st.session_state["cloud_audit_assignment_version"] = assignment_version

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
    if not user:
        _login(store, config)
        return

    if user.get("is_admin"):
        _sidebar(user, "Administration")
        render_admin(
            store,
            _audit_items(CATALOG_VERSION, audit_release_key),
            _eval_items(CATALOG_VERSION, eval_release_key),
        )
        return

    review_state = store.review_state(user["user_id"])
    if review_state is None:
        del st.session_state["annotation_user"]
        st.rerun()
    review_type = next(
        (
            name
            for name in ("audit", "eval")
            if review_state[name]["item_ids"]
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
        render_audit(
            store, user, _audit_items(CATALOG_VERSION, audit_release_key), review_state["audit"]
        )
    else:
        _sidebar(user, "Agent Evaluation")
        render_eval(
            store, user, _eval_items(CATALOG_VERSION, eval_release_key), review_state["eval"]
        )


if __name__ == "__main__":
    main()
