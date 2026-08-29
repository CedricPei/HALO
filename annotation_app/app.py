"""Single-entry Streamlit application for cloud annotation."""

from __future__ import annotations

import hmac
import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from annotation_app.admin_view import render_admin  # noqa: E402
from annotation_app.catalog import (  # noqa: E402
    load_construction_items,
    load_evaluation_items,
)
from annotation_app.config import AppConfig  # noqa: E402
from annotation_app.review_views import (  # noqa: E402
    render_audit,
    render_dashboard,
    render_eval,
)
from annotation_app.storage import AnnotationStore  # noqa: E402
from interfaces.ui import apply_style  # noqa: E402


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
      .login-card {
        background: white;
        border: 1px solid #e1e6ee;
        border-radius: 16px;
        margin: 4rem auto 1rem;
        max-width: 560px;
        padding: 1.4rem 1.7rem .6rem;
      }
      [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding-bottom: .2rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _store(config: AppConfig) -> AnnotationStore:
    return AnnotationStore(config)


@st.cache_data(show_spinner="Loading Benchmark Audit items…")
def _audit_items() -> list[dict]:
    return load_construction_items()


@st.cache_data(show_spinner="Loading Agent Evaluation items…")
def _eval_items() -> list[dict]:
    return load_evaluation_items()


def _login(store: AnnotationStore, config: AppConfig) -> None:
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.title("Proactive Agent Benchmark Annotation Interface")
    st.write("Sign in with the participant ID and access code you received.")
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
                "user_id": "admin",
                "display_name": "Administrator",
                "role": "admin",
            }
        else:
            user = store.authenticate(user_id, access_code)
        if user:
            st.session_state["annotation_user"] = user
            st.rerun()
        st.error("Invalid participant ID or access code.")
    st.markdown("</div>", unsafe_allow_html=True)


def _sidebar(user: dict, config: AppConfig, task_label: str = "") -> None:
    with st.sidebar:
        st.markdown("## Proactive Agent Benchmark Annotation Interface")
        st.caption(f"{user['display_name']} · {user['role'].title()}")
        if task_label:
            st.markdown(f"**Task: {task_label}**")
        st.divider()
        st.caption(config.backend_name)
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
            "Add them to environment variables for local use or to Streamlit "
            "Community Cloud Secrets for deployment. Do not commit secret values to GitHub."
        )
        return
    try:
        store = _store(config)
    except Exception as error:
        st.title("Database connection failed")
        st.error("The App could not connect to its annotation database.")
        st.caption(
            "Check DATABASE_URL. For Supabase on Streamlit Cloud, use the pooled "
            f"connection string. Error type: {type(error).__name__}."
        )
        return

    user = st.session_state.get("annotation_user")
    if not user:
        _login(store, config)
        return

    if user["role"] == "admin":
        _sidebar(user, config, "Administration")
        render_admin(store, config, _audit_items(), _eval_items())
        return

    progress = store.progress_for_user(user["user_id"])
    assigned_types = [
        review_type
        for review_type in ("audit", "eval")
        if progress[review_type]["assigned"]
    ]
    if not assigned_types:
        _sidebar(user, config, "Awaiting assignment")
        render_dashboard(store, user)
        return
    if len(assigned_types) > 1:
        _sidebar(user, config, "Assignment issue")
        st.title("Account assignment needs attention")
        st.error(
            "This account contains both Audit and Eval assignments. Ask the "
            "administrator to issue a single-task account."
        )
        return

    review_type = assigned_types[0]
    if review_type == "audit":
        _sidebar(user, config, "Benchmark Audit")
        render_audit(store, user, _audit_items())
    else:
        _sidebar(user, config, "Agent Evaluation")
        render_eval(store, user, _eval_items())


if __name__ == "__main__":
    main()
