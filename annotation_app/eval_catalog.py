"""Load the frozen human-evaluation release, independent of new model runs."""

import gzip
import hashlib
import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "eval_results"
RELEASE_PATHS = (
    RESULTS_DIR / "human_eval_4000.json.gz",
    RESULTS_DIR / "human_eval_4000_batch2.json.gz",
)
RELEASE_SHA256 = (
    "085f4f13d6bf34f38b65ebae5c661bf2c668648beea1a0cc111100d8553e4ad8",
    "d69d8336a20b0772450156fa96c3e3fcb462b536cd8d6ff9bf9fadf27bc7fc96",
)


def release_signature() -> tuple:
    """Cheap file-version key; do not decompress shared catalogs on reruns."""
    return tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size)
                 for path in RELEASE_PATHS)


def _load_batch(catalog_path: Path, expected_sha256: str):
    raw = catalog_path.read_bytes()
    catalog = json.loads(gzip.decompress(raw))
    items = catalog["items"]
    item_ids = {item["item_id"] for item in items}
    if (hashlib.sha256(raw).hexdigest() != expected_sha256
            or len(items) != 4000 or len(item_ids) != 4000):
        raise ValueError("The human Eval batch differs from the frozen release")
    return items, catalog["subset_id"]


def load_eval_release():
    """Append frozen batches in publication order, keeping existing display IDs."""
    items = []
    seen_items, seen_batches = set(), set()
    for catalog_path, expected_sha256 in zip(RELEASE_PATHS, RELEASE_SHA256):
        batch, subset_id = _load_batch(catalog_path, expected_sha256)
        item_ids = {item["item_id"] for item in batch}
        if seen_items & item_ids or subset_id in seen_batches:
            raise ValueError("Human Eval batches must have distinct items and IDs")
        seen_items.update(item_ids)
        seen_batches.add(subset_id)
        items.extend(batch)
    return items
