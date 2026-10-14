"""Completion page for the retired human annotation platform."""

import streamlit as st


RETIREMENT_VERSION = "completed-2026-10-14-data"

st.set_page_config(page_title="HALO Annotation", page_icon="✓", layout="centered")

# Existing sessions discard the old login state and cached platform resources
# once on entering this version, without connecting to or modifying the database.
if st.session_state.get("_annotation_retirement_version") != RETIREMENT_VERSION:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.session_state["_annotation_retirement_version"] = RETIREMENT_VERSION

st.title("HALO Annotation")
st.info("Human annotation is complete. This platform is now closed.")
st.markdown(
    "The completed Eval and Audit annotations are preserved in the "
    "[HALO repository](https://github.com/CedricPei/HALO/tree/main/annotation_app/data)."
)
