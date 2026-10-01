"""Load the frozen human-evaluation release, independent of new model runs."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).with_name("data")
CATALOG_PATH = DATA_DIR / "human_eval_4000.json.gz"
COHORT_PATH = DATA_DIR / "human_eval_accounts.json"


def load_eval_release():
    raw = CATALOG_PATH.read_bytes()
    catalog = json.loads(gzip.decompress(raw))
    cohort = json.loads(COHORT_PATH.read_text())
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
