"""Infer the evaluation intensity of an Agent's final response."""

from .prompt import INSPECT_COMMANDS


ACTION_INTENSITIES = {
    "email": {
        "flag_email": "L1",
        "create_draft": "L2",
        "send_email": "L3",
    },
    "calendar": {
        "create_reminder": "L1",
        "create_event_draft": "L2",
        "create_event": "L3",
        "update_event": "L3",
        "delete_event": "L3",
    },
    "messaging": {
        "pin_conversation": "L1",
        "create_draft": "L2",
        "send_message": "L3",
    },
    "contacts": {
        "flag_contact": "L1",
        "create_contact": "L2",
        "update_contact": "L2",
    },
    "wallet": {
        "flag_transaction": "L1",
        "create_draft": "L2",
        "pay": "L3",
        "refund_transaction": "L3",
    },
}
INTENSITY_LEVELS = ("L0", "L1", "L2", "L3")


def action_intensity(content):
    """Infer final intensity from no action or one action command."""
    if content is None:
        return "L0"
    if not isinstance(content, dict):
        raise ValueError("final content must be null or exactly one action command")
    if (content.get("app"), content.get("function")) in INSPECT_COMMANDS:
        return "L0"
    try:
        return ACTION_INTENSITIES[content["app"]][content["function"]]
    except KeyError as error:
        raise ValueError(
            "final content is not an evaluated-Agent action"
        ) from error
