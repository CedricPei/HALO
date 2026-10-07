"""Load the frozen human-evaluation release, independent of new model runs."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).with_name("data")
CATALOG_PATH = DATA_DIR / "human_eval_4000.json.gz"
COHORT_PATH = DATA_DIR / "human_eval_accounts.json"
RELEASE_PATHS = (
    (CATALOG_PATH, COHORT_PATH),
    (DATA_DIR / "human_eval_4000_batch2.json.gz",
     DATA_DIR / "human_eval_accounts_batch2.json"),
)


def release_signature() -> tuple:
    """Cheap file-version key; do not decompress shared catalogs on reruns."""
    return tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size)
                 for pair in RELEASE_PATHS for path in pair)


def _load_batch(catalog_path: Path, cohort_path: Path):
    raw = catalog_path.read_bytes()
    catalog = json.loads(gzip.decompress(raw))
    cohort = json.loads(cohort_path.read_text())
    items = catalog["items"]
    item_ids = {item["item_id"] for item in items}
    assigned = [item_id for account in cohort["accounts"] for item_id in account["item_ids"]]
    if (hashlib.sha256(raw).hexdigest() != cohort["catalog_sha256"]
            or catalog["subset_id"] != cohort["subset_id"]
            or len(items) != 4000 or len(item_ids) != 4000
            or Counter(assigned) != Counter(item_ids)
            or len(cohort["accounts"]) != 20
            or len({a["user_id"] for a in cohort["accounts"]}) != 20
            or any(len(a["item_ids"]) != 200 for a in cohort["accounts"])):
        raise ValueError("The human Eval catalog and account assignments do not match")
    return items, cohort


def load_eval_release():
    """Append frozen batches in publication order, keeping existing display IDs."""
    items, cohorts = [], []
    seen_items, seen_users, seen_batches = set(), set(), set()
    for catalog_path, cohort_path in RELEASE_PATHS:
        batch, cohort = _load_batch(catalog_path, cohort_path)
        item_ids = {item["item_id"] for item in batch}
        user_ids = {account["user_id"] for account in cohort["accounts"]}
        if (seen_items & item_ids or seen_users & user_ids
                or cohort["subset_id"] in seen_batches):
            raise ValueError("Human Eval batches must have distinct items, accounts and IDs")
        seen_items.update(item_ids)
        seen_users.update(user_ids)
        seen_batches.add(cohort["subset_id"])
        items.extend(batch)
        cohorts.append(cohort)
    return items, cohorts
