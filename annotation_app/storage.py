"""Portable SQLite/PostgreSQL storage for current annotation answers."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from annotation_app.config import AppConfig


USER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
ACCESS_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_user_id(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    if not USER_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "User ID must be 3-64 characters using lowercase letters, numbers, ., _, or -."
        )
    return normalized


def generate_access_code() -> str:
    return "-".join(
        "".join(secrets.choice(ACCESS_CODE_ALPHABET) for _ in range(4))
        for _ in range(3)
    )


class AnnotationStore:
    """Central review store; original benchmark files are never mutated."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.sqlite = config.database_url.startswith("sqlite:///")
        if self.sqlite:
            self.sqlite_path = Path(config.database_url.removeprefix("sqlite:///"))
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    @contextmanager
    def _connect(self):
        if not self.sqlite:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as error:
                raise RuntimeError(
                    "PostgreSQL mode requires the psycopg package."
                ) from error
            with psycopg.connect(
                self.config.database_url,
                row_factory=dict_row,
                prepare_threshold=None,
            ) as connection:
                yield connection
            return

        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _sql(self, statement: str) -> str:
        return statement if self.sqlite else statement.replace("?", "%s")

    def ensure_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        statements = [
            statement.strip()
            for statement in schema_path.read_text(encoding="utf-8").split(";")
            if statement.strip()
        ]
        with self._connect() as connection:
            for statement in statements:
                if self.sqlite and statement.upper().startswith("ALTER TABLE"):
                    continue
                connection.execute(statement)
            self._migrate_legacy_revisions(connection)

    def _migrate_legacy_revisions(self, connection) -> None:
        """Keep the latest legacy answer per item, then remove its history table."""
        if self.sqlite:
            exists = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='annotation_revisions'"
            ).fetchone()
        else:
            exists = connection.execute(
                "SELECT to_regclass('public.annotation_revisions') AS table_name"
            ).fetchone()["table_name"]
        if not exists:
            return

        rows = connection.execute(
            "SELECT revision_id, user_id, review_type, item_id, payload_json, "
            "created_at FROM annotation_revisions "
            "ORDER BY created_at, revision_id"
        ).fetchall()
        latest = {}
        for row in rows:
            latest[(row["user_id"], row["review_type"], row["item_id"])] = row
        for row in latest.values():
            connection.execute(
                self._sql(
                    "INSERT INTO annotations "
                    "(annotation_id, user_id, review_type, item_id, payload_json, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(user_id, review_type, item_id) DO UPDATE SET "
                    "annotation_id=excluded.annotation_id, "
                    "payload_json=excluded.payload_json, "
                    "created_at=excluded.created_at"
                ),
                (
                    row["revision_id"],
                    row["user_id"],
                    row["review_type"],
                    row["item_id"],
                    row["payload_json"],
                    row["created_at"],
                ),
            )
        connection.execute("DROP TABLE annotation_revisions")

    def _hash_code(self, access_code: str) -> str:
        if not self.config.auth_pepper:
            raise RuntimeError("AUTH_PEPPER must be configured before creating users.")
        return hmac.new(
            self.config.auth_pepper.encode("utf-8"),
            access_code.strip().upper().encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def create_user(
        self,
        user_id: str,
        access_code: str,
    ) -> None:
        user_id = normalize_user_id(user_id)
        if user_id == self.config.admin_username.strip().lower():
            raise ValueError("This user ID is reserved for the bootstrap administrator.")
        with self._connect() as connection:
            connection.execute(
                self._sql(
                    "INSERT INTO annotation_users "
                    "(user_id, access_code_hash, created_at) VALUES (?, ?, ?)"
                ),
                (
                    user_id,
                    self._hash_code(access_code),
                    utc_now(),
                ),
            )

    def authenticate(self, user_id: str, access_code: str) -> dict | None:
        try:
            user_id = normalize_user_id(user_id)
        except ValueError:
            return None
        with self._connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT user_id, access_code_hash FROM annotation_users "
                    "WHERE user_id=?"
                ),
                (user_id,),
            ).fetchone()
            if not row:
                return None
            if not hmac.compare_digest(
                row["access_code_hash"], self._hash_code(access_code)
            ):
                return None
            connection.execute(
                self._sql(
                    "UPDATE annotation_users SET last_login_at=? WHERE user_id=?"
                ),
                (utc_now(), user_id),
            )
        return {"user_id": row["user_id"]}

    def list_users(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT user_id, created_at, last_login_at "
                "FROM annotation_users ORDER BY created_at, user_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def reset_access_code(self, user_id: str, access_code: str) -> None:
        with self._connect() as connection:
            connection.execute(
                self._sql(
                    "UPDATE annotation_users SET access_code_hash=? WHERE user_id=?"
                ),
                (self._hash_code(access_code), normalize_user_id(user_id)),
            )

    def delete_user(self, user_id: str) -> None:
        user_id = normalize_user_id(user_id)
        with self._connect() as connection:
            for table in (
                "annotations",
                "annotation_assignments",
                "annotation_users",
            ):
                connection.execute(
                    self._sql(f"DELETE FROM {table} WHERE user_id=?"),
                    (user_id,),
                )

    def assign_tasks(
        self,
        user_id: str,
        review_type: str,
        item_ids: Iterable[str],
    ) -> None:
        user_id = normalize_user_id(user_id)
        if review_type not in {"audit", "eval"}:
            raise ValueError("Invalid review type.")
        item_ids = list(dict.fromkeys(item_ids))
        if not item_ids:
            return
        now = utc_now()
        with self._connect() as connection:
            for position, item_id in enumerate(item_ids):
                connection.execute(
                    self._sql(
                        "INSERT INTO annotation_assignments "
                        "(user_id, review_type, item_id, position, status, assigned_at) "
                        "VALUES (?, ?, ?, ?, 'assigned', ?)"
                    ),
                    (user_id, review_type, item_id, position, now),
                )

    def assignment_ids(self, user_id: str, review_type: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                self._sql(
                    "SELECT item_id FROM annotation_assignments "
                    "WHERE user_id=? AND review_type=? ORDER BY position, item_id"
                ),
                (user_id, review_type),
            ).fetchall()
        return [row["item_id"] for row in rows]

    def reviewed_ids(self, user_id: str, review_type: str) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                self._sql(
                    "SELECT item_id FROM annotation_assignments "
                    "WHERE user_id=? AND review_type=? AND status='submitted'"
                ),
                (user_id, review_type),
            ).fetchall()
        return {row["item_id"] for row in rows}

    def submitted_users_by_item(self, review_type: str) -> dict[str, set[str]]:
        with self._connect() as connection:
            rows = connection.execute(
                self._sql(
                    "SELECT item_id, user_id FROM annotation_assignments "
                    "WHERE review_type=? AND status='submitted'"
                ),
                (review_type,),
            ).fetchall()
        result: dict[str, set[str]] = {}
        for row in rows:
            result.setdefault(row["item_id"], set()).add(row["user_id"])
        return result

    def replace_pending_assignments(
        self,
        review_type: str,
        plan: dict[str, list[str]],
    ) -> None:
        """Keep submitted work and atomically replace all unfinished assignments."""
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                self._sql(
                    "DELETE FROM annotation_assignments "
                    "WHERE review_type=? AND status='assigned'"
                ),
                (review_type,),
            )
            for user_id, item_ids in plan.items():
                row = connection.execute(
                    self._sql(
                        "SELECT COALESCE(MAX(position), -1) AS maximum "
                        "FROM annotation_assignments "
                        "WHERE user_id=? AND review_type=?"
                    ),
                    (user_id, review_type),
                ).fetchone()
                start = int(row["maximum"]) + 1
                for offset, item_id in enumerate(item_ids):
                    connection.execute(
                        self._sql(
                            "INSERT INTO annotation_assignments "
                            "(user_id, review_type, item_id, position, status, "
                            "assigned_at) VALUES (?, ?, ?, ?, 'assigned', ?)"
                        ),
                        (user_id, review_type, item_id, start + offset, now),
                    )

    def get_annotation(
        self, user_id: str, review_type: str, item_id: str
    ) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT annotation_id, payload_json, created_at "
                    "FROM annotations "
                    "WHERE user_id=? AND review_type=? AND item_id=? "
                    "LIMIT 1"
                ),
                (user_id, review_type, item_id),
            ).fetchone()
            if not row:
                return None
        return {
            "annotation_id": row["annotation_id"],
            "payload": json.loads(row["payload_json"]),
            "created_at": row["created_at"],
        }

    def save_annotation(
        self,
        user_id: str,
        review_type: str,
        item_id: str,
        payload: dict,
    ) -> str:
        if review_type not in {"audit", "eval"}:
            raise ValueError("Invalid review type.")
        annotation_id = str(uuid.uuid4())
        now = utc_now()
        with self._connect() as connection:
            assigned = connection.execute(
                self._sql(
                    "SELECT 1 FROM annotation_assignments "
                    "WHERE user_id=? AND review_type=? AND item_id=?"
                ),
                (user_id, review_type, item_id),
            ).fetchone()
            if not assigned:
                raise ValueError("This item is not assigned to the current user.")
            connection.execute(
                self._sql(
                    "INSERT INTO annotations "
                    "(annotation_id, user_id, review_type, item_id, payload_json, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(user_id, review_type, item_id) DO UPDATE SET "
                    "annotation_id=excluded.annotation_id, "
                    "payload_json=excluded.payload_json, "
                    "created_at=excluded.created_at"
                ),
                (
                    annotation_id,
                    user_id,
                    review_type,
                    item_id,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
            connection.execute(
                self._sql(
                    "UPDATE annotation_assignments "
                    "SET status='submitted', submitted_at=? "
                    "WHERE user_id=? AND review_type=? AND item_id=?"
                ),
                (now, user_id, review_type, item_id),
            )
        return annotation_id

    def progress_for_user(self, user_id: str) -> dict[str, dict[str, int]]:
        progress = {
            "audit": {"assigned": 0, "submitted": 0},
            "eval": {"assigned": 0, "submitted": 0},
        }
        with self._connect() as connection:
            rows = connection.execute(
                self._sql(
                    "SELECT review_type, COUNT(*) AS assigned, "
                    "SUM(CASE WHEN status='submitted' THEN 1 ELSE 0 END) AS submitted "
                    "FROM annotation_assignments WHERE user_id=? GROUP BY review_type"
                ),
                (user_id,),
            ).fetchall()
        for row in rows:
            progress[row["review_type"]] = {
                "assigned": int(row["assigned"]),
                "submitted": int(row["submitted"] or 0),
            }
        return progress

    def progress_all_users(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT u.user_id, a.review_type, COUNT(a.item_id) AS assigned, "
                "SUM(CASE WHEN a.status='submitted' THEN 1 ELSE 0 END) AS submitted "
                "FROM annotation_users u "
                "LEFT JOIN annotation_assignments a ON a.user_id=u.user_id "
                "GROUP BY u.user_id, a.review_type "
                "ORDER BY u.user_id, a.review_type"
            ).fetchall()
        return [dict(row) for row in rows]

    def export_annotations(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT annotation_id, user_id, review_type, item_id, payload_json, "
                "created_at FROM annotations "
                "ORDER BY user_id, review_type, item_id"
            ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["payload"] = json.loads(record.pop("payload_json"))
            records.append(record)
        return records
