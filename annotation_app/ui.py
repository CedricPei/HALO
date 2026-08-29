"""Styling and rendering helpers for the cloud annotation interface."""

import html
from datetime import datetime

import streamlit as st


STYLE = """
<style>
    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .block-container {
        box-sizing: border-box;
        max-width: none;
        width: 100%;
        padding: 1.9rem clamp(1.25rem, 2.5vw, 3rem) 2.5rem;
    }
    [data-testid="stVerticalBlock"] {gap: .38rem;}
    [data-testid="stHorizontalBlock"]:has(.st-key-evidence-pane) {
        align-items: stretch;
    }
    [data-testid="stHorizontalBlock"]:has(.st-key-evidence-pane)
        > [data-testid="stColumn"]:first-child {
        position: relative;
    }
    [data-testid="stHorizontalBlock"]:has(.st-key-evidence-pane)
        > [data-testid="stColumn"]:first-child
        > [data-testid="stVerticalBlockBorderWrapper"] {
        height: 100% !important;
        inset: 0;
        position: absolute;
        width: 100%;
    }
    [data-testid="stHorizontalBlock"]:has(.st-key-evidence-pane)
        > [data-testid="stColumn"]:first-child
        > [data-testid="stVerticalBlockBorderWrapper"]
        > [data-testid="stVerticalBlock"],
    [data-testid="stVerticalBlockBorderWrapper"]:has(> .st-key-evidence-pane) {
        height: 100% !important;
    }
    [data-testid="stHorizontalBlock"]:has(.st-key-evidence-pane)
        > [data-testid="stColumn"]:nth-child(2) li {
        line-height: 1.35;
        padding-left: .6rem !important;
        position: relative;
    }
    [data-testid="stHorizontalBlock"]:has(.st-key-evidence-pane)
        > [data-testid="stColumn"]:nth-child(2) ul {
        list-style: none;
        padding-left: 0 !important;
    }
    [data-testid="stHorizontalBlock"]:has(.st-key-evidence-pane)
        > [data-testid="stColumn"]:nth-child(2) li::before {
        content: "•";
        left: 0;
        position: absolute;
    }
    [data-testid="stSidebar"] {border-right: 1px solid #e4e8ef;}
    [data-testid="stForm"] {border: 0; padding: 0;}
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #aeb8c7 !important;
        box-shadow: 0 1px 2px rgba(16, 24, 40, .05);
        font-size: .94rem !important;
        height: 1.75rem !important;
        min-height: 1.75rem !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div > div {
        padding-bottom: 0 !important;
        padding-top: 0 !important;
        transform: translateY(.075rem);
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:first-child {
        padding-left: .55rem !important;
        padding-right: .35rem !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:last-child {
        padding-left: .25rem !important;
        padding-right: .45rem !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] span {
        font-size: .94rem !important;
        line-height: 1.3rem !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color: #667085 !important;
    }
    [data-testid="stExpander"] {
        background: #ffffff;
        border-color: #e1e6ee;
        border-radius: 10px;
    }
    [data-testid="stExpander"] details:not([open]) summary {
        padding: .25rem .75rem;
    }
    [data-testid="stExpander"] summary > span,
    [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] {
        width: 100%;
    }
    [data-testid="stExpander"] summary > span {
        flex: 1;
    }
    [data-testid="stExpander"] summary p {
        align-items: center;
        display: flex;
        justify-content: space-between;
        width: 100%;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: #e1e6ee;
        border-radius: 12px;
    }
    h1 {
        color: #172033 !important;
        font-size: 2.15rem !important;
        font-weight: 680 !important;
        letter-spacing: -.025em;
        line-height: 1.16 !important;
        margin-bottom: .2rem !important;
    }
    h4 {
        color: #a16207 !important;
        font-size: 1.16rem !important;
        font-weight: 650 !important;
        letter-spacing: -.01em;
        margin-top: .45rem !important;
        margin-bottom: .08rem !important;
    }
    h5 {
        color: #dc2626 !important;
        font-size: .96rem !important;
        font-weight: 700 !important;
        letter-spacing: -.005em;
        margin-top: .8rem !important;
        margin-bottom: .08rem !important;
    }
    [data-testid="stElementContainer"]:has(h5#verdict)
        + [data-testid="stElementContainer"]:has([data-testid="stRadio"]) {
        margin-top: -.3rem;
    }
    p, li {font-size: .94rem; line-height: 1.55;}
    [data-testid="stMarkdownContainer"] p {margin-bottom: .35rem;}
    [data-testid="stMarkdownContainer"] ul {margin-bottom: .35rem; margin-top: .2rem;}
    [data-testid="stCaptionContainer"] p {font-size: .8rem; line-height: 1.45;}
    .structured-summary {
        list-style: none;
        margin: .05rem 0 .45rem;
        padding-left: 0 !important;
    }
    .structured-item {
        color: #344054;
        font-size: .94rem;
        line-height: 1.48;
        margin: 0 0 .24rem;
        padding-left: .6rem !important;
        position: relative;
    }
    .structured-item::before {
        content: "•";
        left: 0;
        position: absolute;
    }
    .structured-item strong {color: #1d2939; font-weight: 680;}
    .requirement-group {margin: 0;}
    .requirement-group strong {
        color: inherit;
        display: block;
        font-weight: 700;
        line-height: 1.25;
    }
    .requirement-group ul {
        margin-top: 0 !important;
        margin-bottom: .08rem !important;
    }
    [data-testid="stCheckbox"] label[data-baseweb="checkbox"] > span {
        border-radius: 50% !important;
    }
    [data-testid="stCheckbox"] {margin-bottom: -.6rem;}
    .rejection-heading {
        color: #dc2626;
        font-size: .96rem;
        font-weight: 700;
        line-height: 1.2;
        margin: -.45rem 0 .08rem;
    }
    .eval-actions-spacer {height: 1.15rem;}
    .review-meta {
        color: #667085;
        display: flex;
        flex-wrap: wrap;
        font-size: .8rem;
        gap: .3rem 1.1rem;
        margin: 0 0 1rem;
    }
    .review-meta strong {color: #475467; font-weight: 600;}
    .intensity-inference {
        color: #1d2939;
        font-size: .94rem;
        font-weight: 500;
        margin: .85rem 0 0;
        padding-bottom: .65rem;
    }
    .intensity-inference strong {color: #1d2939; font-weight: 700;}
    .review-heading {padding-bottom: 1.1rem;}
    .evaluation-heading {margin-top: .55rem;}
    .review-heading h4 {
        line-height: 1.2 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .review-heading h5 {
        line-height: 1.2 !important;
        margin: .28rem 0 0 !important;
        padding: 0 !important;
    }
</style>
"""


def apply_style():
    st.markdown(STYLE, unsafe_allow_html=True)


def format_event_time(value):
    """Render an ISO Event time without the benchmark year."""
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    clock = moment.strftime("%I:%M %p").lstrip("0")
    return f"{moment.strftime('%b')} {moment.day} {clock}"


def _compact(value, limit=62):
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[:limit - 1].rstrip()}…"


def event_summary(event):
    """Extract a readable cue from the command's most informative parameter."""
    parameter = event.get("parameter", {})
    name = (event.get("app"), event.get("function"))
    if name == ("email", "create_email") and parameter.get("subject"):
        return f"Email received — {_compact(parameter['subject'])}"
    if name == ("messaging", "create_message") and parameter.get("content"):
        return f"Message received — {_compact(parameter['content'])}"
    if name == ("calendar", "create_event") and parameter.get("title"):
        return f"Calendar added — {_compact(parameter['title'])}"
    if name == ("calendar", "update_event") and parameter.get("title"):
        return f"Calendar updated — {_compact(parameter['title'])}"
    if name == ("contacts", "update_contact") and parameter.get("name"):
        return f"Contact updated — {_compact(parameter['name'])}"
    return f"{event['app']}.{event['function']}"


def format_event_label(event_id, event):
    event_number = event_id.removeprefix("event")
    title = f"Event {event_number}: {event_summary(event)}"
    if event.get("time"):
        return f"**{title}** {format_event_time(event['time'])}"
    return f"**{title}**"


def render_structured_text(value):
    """Render colon-delimited summary lines, with a paragraph fallback."""
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    items = [line.partition(":") for line in lines]
    if len(items) < 2 or any(not separator or not label or not body for label, separator, body in items):
        st.write(value)
        return
    rendered = "".join(
        '<li class="structured-item">'
        f"<strong>{html.escape(label.strip())}:</strong> "
        f"{html.escape(body.strip())}</li>"
        for label, _, body in items
    )
    st.markdown(
        f'<ul class="structured-summary">{rendered}</ul>',
        unsafe_allow_html=True,
    )


def metadata(**values):
    rendered = "".join(
        f"<span><strong>{html.escape(label)}</strong> {html.escape(str(value))}</span>"
        for label, value in values.items()
    )
    st.markdown(f'<div class="review-meta">{rendered}</div>', unsafe_allow_html=True)
