"""Shared helpers for the JSON-backed apps."""

import copy
import json
from datetime import datetime
from pathlib import Path


class JsonStore:
    """Load one JSON state file and save it in a readable format."""

    def __init__(self, path, empty_state):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = empty_state
            self.save()

    def save(self):
        """Write the current in-memory state to disk."""
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def clone(value):
    """Return data that callers can change without mutating stored state."""
    return copy.deepcopy(value)


def public(value):
    """Copy stored data while hiding internal ID fields."""
    if isinstance(value, list):
        return [public(item) for item in value]
    if isinstance(value, dict):
        return {
            key: public(item)
            for key, item in value.items()
            if not key.endswith("_id")
        }
    return copy.deepcopy(value)


def next_id(items, id_field, prefix):
    """Generate the next internal prefix_000001 style ID."""
    numbers = []
    for item in items:
        value = item.get(id_field, "")
        if value.startswith(prefix + "_") and value[len(prefix) + 1 :].isdigit():
            numbers.append(int(value[len(prefix) + 1 :]))
    return f"{prefix}_{max(numbers, default=0) + 1:06d}"


def parse_time(value):
    """Parse an ISO 8601 timestamp, including the common Z suffix."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def in_time_range(value, start_time=None, end_time=None):
    """Check whether a timestamp is inside an inclusive range."""
    value = parse_time(value)
    return (start_time is None or value >= parse_time(start_time)) and (
        end_time is None or value <= parse_time(end_time)
    )


def keyword_match(query, *values):
    """Run a case-insensitive substring match across strings and lists."""
    query = query.casefold()
    text = []
    for value in values:
        text.extend(value if isinstance(value, list) else [value or ""])
    return any(query in str(value).casefold() for value in text)
