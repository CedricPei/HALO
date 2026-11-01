"""Optional platform samples, independent of exported human judgments."""

import gzip
import json
import os
from pathlib import Path


def source_paths() -> dict[str, tuple[Path, ...]]:
    directory = Path(os.getenv("ANNOTATION_DATA_DIR") or Path(__file__).with_name("data"))
    return {
        name: tuple(directory / f"{name}-items{suffix}" for suffix in (".json", ".json.gz"))
        for name in ("audit", "eval")
    }


def source_signature() -> tuple:
    """Invalidate the shared cache only when a sample file changes or disappears."""
    return tuple(
        (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
        if path.is_file() else (str(path.resolve()), None, None)
        for paths in source_paths().values() for path in paths
    )


def load_items(review_type: str) -> list[dict]:
    paths = source_paths()[review_type]
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        return []
    raw = path.read_bytes()
    payload = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Annotation samples must be a list or an object containing items")
    if any(not isinstance(item, dict) or not item.get("item_id") for item in items):
        raise ValueError("Every annotation sample must have an item_id")
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("Annotation sample IDs must be unique")
    return items
