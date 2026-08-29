"""Import a legacy local reviews.sqlite file into the central app store."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from annotation_app.config import AppConfig
from annotation_app.storage import AnnotationStore


def _rows(connection, table: str) -> list[dict]:
    connection.row_factory = sqlite3.Row
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not exists:
        return []
    return [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]


def import_reviews(source: Path, user_id: str, store: AnnotationStore) -> dict:
    imported = skipped = 0
    with sqlite3.connect(source) as connection:
        audit_rows = _rows(connection, "construction_reviews")
        eval_rows = _rows(connection, "evaluation_reviews")

    for review_type, rows in (("audit", audit_rows), ("eval", eval_rows)):
        assigned = set(store.assignment_ids(user_id, review_type))
        for row in rows:
            if row["item_id"] not in assigned:
                skipped += 1
                continue
            if review_type == "audit":
                payload = {
                    "verdict": row["verdict"],
                    "comment": row.get("issues", ""),
                }
            else:
                payload = {
                    "acceptable": row["acceptable"],
                    "primary_reason": row.get("primary_reason", ""),
                    "timing": row.get("timing", ""),
                    "intensity": row.get("intensity", ""),
                    "diagnostics": json.loads(row.get("diagnostics_json") or "{}"),
                    "comments": row.get("comments", ""),
                }
            current = store.get_annotation(user_id, review_type, row["item_id"])
            if current and current["payload"] == payload:
                skipped += 1
                continue
            store.save_annotation(user_id, review_type, row["item_id"], payload)
            imported += 1
    return {"imported": imported, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--user-id", required=True)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source does not exist: {args.source}")
    store = AnnotationStore(AppConfig.load())
    result = import_reviews(args.source, args.user_id, store)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
