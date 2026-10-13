"""Assign the second frozen Eval batch to eval21–eval40 without touching batch one."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from annotation_app.storage import generate_access_code, hash_admin_password
from benchmark_catalog import ROOT_DIR
from evaluation.human_subset import BATCH2_FILENAME, BATCH2_SUBSET_ID


def build_cohort(catalog: dict, previous: dict, previous_cohort: dict,
                 catalog_sha256: str) -> tuple[dict, list[dict]]:
    """Reuse the published scenario partition; randomize only new access codes."""
    old_items = {item["item_id"]: item for item in previous["items"]}
    by_scenario = defaultdict(list)
    for item in catalog["items"]:
        if item["item_id"] in old_items:
            raise ValueError("The second batch overlaps the first batch")
        by_scenario[item["source_path"]].append(item)
    if (catalog["subset_id"] != BATCH2_SUBSET_ID
            or len(catalog["items"]) != 4000 or len(by_scenario) != 400
            or len(previous_cohort["accounts"]) != 20):
        raise ValueError("Expected two frozen 4,000-item batches and 20 existing accounts")

    accounts, credentials, assigned = [], [], []
    for number, old_account in enumerate(previous_cohort["accounts"], start=1):
        if old_account["user_id"] != f"eval{number:02d}":
            raise ValueError("The published account template must be eval01–eval20")
        scenarios = list(dict.fromkeys(
            old_items[item_id]["source_path"] for item_id in old_account["item_ids"]
        ))
        items = [item for source in scenarios for item in by_scenario[source]]
        if (len(scenarios) != 20 or len(items) != 200
                or Counter(x["model"] for x in items)
                   != Counter({model: 40 for model in catalog["models"]})
                or sorted(Counter(x["domain_id"] for x in items).values()) != [20] * 10
                or sorted(Counter(x["trigger_type"] for x in items).values()) != [50] * 4):
            raise ValueError("The account template does not satisfy the balanced quotas")
        user_id = f"eval{number + 20:02d}"
        access_code = generate_access_code() + "-" + generate_access_code()
        item_ids = [item["item_id"] for item in items]
        accounts.append({"user_id": user_id,
                         "access_code_hash": hash_admin_password(access_code),
                         "item_ids": item_ids})
        credentials.append({"user_id": user_id, "access_code": access_code})
        assigned.extend(item_ids)
    if Counter(assigned) != Counter(x["item_id"] for x in catalog["items"]):
        raise ValueError("Every second-batch item must be assigned exactly once")
    return {
        "schema_version": 1, "subset_id": catalog["subset_id"],
        "catalog_sha256": catalog_sha256,
        "reviews_per_item": 1, "items_per_account": 200,
        "assignment_template_subset_id": previous["subset_id"],
        "accounts": accounts,
    }, credentials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-output", type=Path, required=True,
                        help="New private text file outside the repository")
    parser.add_argument("--app-url", default=os.getenv("HALO_ANNOTATION_URL", ""),
                        help="Annotation deployment URL to include in the handout")
    args = parser.parse_args()
    credentials_path = args.credentials_output.expanduser().resolve()
    if credentials_path == ROOT_DIR or ROOT_DIR in credentials_path.parents:
        parser.error("Deliver plaintext credentials outside the repository")
    data_dir = ROOT_DIR / "annotation_app/data"
    manifest_path = data_dir / "human_eval_accounts_batch2.json"
    if manifest_path.exists() or credentials_path.exists():
        parser.error("Refusing to replace existing accounts or access codes")
    raw = (data_dir / BATCH2_FILENAME).read_bytes()
    catalog = json.loads(gzip.decompress(raw))
    previous = json.loads(gzip.decompress((data_dir / "human_eval_4000.json.gz").read_bytes()))
    previous_cohort = json.loads((data_dir / "human_eval_accounts.json").read_text())
    cohort, credentials = build_cohort(catalog, previous, previous_cohort,
                                     hashlib.sha256(raw).hexdigest())
    # Save the only plaintext copy first, with owner-only permissions.
    fd = os.open(credentials_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("HALO — Eval batch 2\n")
        if args.app_url:
            handle.write(args.app_url.strip() + "\n")
        handle.write("Each account has 200 assigned items.\n\nParticipant ID\tAccess code\n")
        for entry in credentials:
            handle.write(f"{entry['user_id']}\t{entry['access_code']}\n")
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(cohort, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"subset_id": cohort["subset_id"], "accounts": len(credentials),
                      "manifest": str(manifest_path), "credentials": str(credentials_path)}))


if __name__ == "__main__":
    main()
