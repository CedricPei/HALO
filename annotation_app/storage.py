"""Portable SQLite/PostgreSQL storage with append-only annotation revisions."""

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


def generate_access_code(groups: int = 3, width: int = 4) -> str:
    return "-".join(
        "".join(secrets.choice(ACCESS_CODE_ALPHABET) for _ in range(width))
        for _ in range(groups)
    )


class AnnotationStore:
    """Central review store; original benchmark files are never mutated."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.postgres = bool(config.database_url)
        if not self.postgres:
            config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    @contextmanager
    def _connect(self):
        if self.postgres:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as error:
                raise RuntimeError(
                    "PostgreSQL mode requires psycopg; install annotation_app/requirements.txt."
                ) from error
            with psycopg.connect(
                self.config.database_url,
                row_factory=dict_row,
                prepare_threshold=None,
            ) as connection:
                yield connection
            return

        connection = sqlite3.connect(self.config.sqlite_path)
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
        return statement.replace("?", "%s") if self.postgres else statement

    def ensure_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        statements = [
            statement.strip()
            for statement in schema_path.read_text(encoding="utf-8").split(";")
            if statement.strip()
        ]
        with self._connect() as connection:
            for statement in statements:
                if not self.postgres and statement.upper().startswith("ALTER TABLE"):
                    continue
                connection.execute(statement)

    def _hash_code(self, access_code: str) -> str:
        if not self.config.auth_pepper:
            raise RuntimeError("AUTH_PEPPER must be configured before creating users.")
        return hmac.new(
            self.config.auth_pepper.encode("utf-8"),
            access_code.strip().upper().encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def log_admin_event(self, actor_id: str, event_type: str, details: dict) -> None:
        with self._connect() as connection:
            connection.execute(
                self._sql(
                    "INSERT INTO annotation_admin_events "
                    "(event_id, actor_id, event_type, details_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)"
                ),
                (
                    str(uuid.uuid4()),
                    actor_id,
                    event_type,
                    json.dumps(details, ensure_ascii=False, sort_keys=True),
                    utc_now(),
                ),
            )

    def create_user(
        self,
        user_id: str,
        display_name: str,
        role: str,
        access_code: str,
    ) -> dict:
        user_id = normalize_user_id(user_id)
        if user_id == self.config.admin_username.strip().lower():
            raise ValueError("This user ID is reserved for the bootstrap administrator.")
        display_name = display_name.strip()
        if not display_name:
            raise ValueError("Display name is required.")
        if role not in {"annotator", "reviewer", "admin"}:
            raise ValueError("Invalid role.")
        with self._connect() as connection:
            connection.execute(
                self._sql(
                    "INSERT INTO annotation_users "
                    "(user_id, display_name, role, access_code_hash, active, created_at) "
                    "VALUES (?, ?, ?, ?, 1, ?)"
                ),
                (
                    user_id,
                    display_name,
                    role,
                    self._hash_code(access_code),
                    utc_now(),
                ),
            )
        return {"user_id": user_id, "display_name": display_name, "role": role}

    def authenticate(self, user_id: str, access_code: str) -> dict | None:
        try:
            user_id = normalize_user_id(user_id)
        except ValueError:
            return None
        with self._connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT user_id, display_name, role, access_code_hash, active "
                    "FROM annotation_users WHERE user_id=?"
                ),
                (user_id,),
            ).fetchone()
            if not row or not int(row["active"]):
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
        return {
            "user_id": row["user_id"],
            "display_name": row["display_name"],
            "role": row["role"],
        }

    def list_users(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT user_id, display_name, role, active, created_at, last_login_at "
                "FROM annotation_users ORDER BY created_at, user_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def set_user_active(self, user_id: str, active: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                self._sql("UPDATE annotation_users SET active=? WHERE user_id=?"),
                (1 if active else 0, normalize_user_id(user_id)),
            )

    def reset_access_code(self, user_id: str, access_code: str) -> None:
        with self._connect() as connection:
            connection.execute(
                self._sql(
                    "UPDATE annotation_users SET access_code_hash=? WHERE user_id=?"
                ),
                (self._hash_code(access_code), normalize_user_id(user_id)),
            )

    def assign_tasks(
        self,
        user_id: str,
        review_type: str,
        item_ids: Iterable[str],
    ) -> int:
        user_id = normalize_user_id(user_id)
        if review_type not in {"audit", "eval"}:
            raise ValueError("Invalid review type.")
        item_ids = list(dict.fromkeys(item_ids))
        if not item_ids:
            return 0
        now = utc_now()
        inserted = 0
        with self._connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT COALESCE(MAX(position), -1) AS maximum "
                    "FROM annotation_assignments WHERE user_id=? AND review_type=?"
                ),
                (user_id, review_type),
            ).fetchone()
            next_position = int(row["maximum"]) + 1
            for item_id in item_ids:
                cursor = connection.execute(
                    self._sql(
                        "INSERT INTO annotation_assignments "
                        "(user_id, review_type, item_id, position, status, assigned_at) "
                        "VALUES (?, ?, ?, ?, 'assigned', ?) "
                        "ON CONFLICT(user_id, review_type, item_id) DO NOTHING"
                    ),
                    (user_id, review_type, item_id, next_position, now),
                )
                if cursor.rowcount:
                    inserted += 1
                    next_position += 1
        return inserted

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

    def latest_annotation(
        self, user_id: str, review_type: str, item_id: str
    ) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT revision_id, payload_json, state, created_at "
                    "FROM annotation_revisions "
                    "WHERE user_id=? AND review_type=? AND item_id=? "
                    "ORDER BY created_at DESC, revision_id DESC LIMIT 1"
                ),
                (user_id, review_type, item_id),
            ).fetchone()
            if not row:
                return None
            count_row = connection.execute(
                self._sql(
                    "SELECT COUNT(*) AS count FROM annotation_revisions "
                    "WHERE user_id=? AND review_type=? AND item_id=?"
                ),
                (user_id, review_type, item_id),
            ).fetchone()
        return {
            "revision_id": row["revision_id"],
            "payload": json.loads(row["payload_json"]),
            "state": row["state"],
            "created_at": row["created_at"],
            "version": int(count_row["count"]),
        }

    def save_annotation(
        self,
        user_id: str,
        review_type: str,
        item_id: str,
        payload: dict,
        state: str = "submitted",
    ) -> str:
        if review_type not in {"audit", "eval"}:
            raise ValueError("Invalid review type.")
        if state not in {"draft", "submitted"}:
            raise ValueError("Invalid revision state.")
        revision_id = str(uuid.uuid4())
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
                    "INSERT INTO annotation_revisions "
                    "(revision_id, user_id, review_type, item_id, payload_json, state, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    revision_id,
                    user_id,
                    review_type,
                    item_id,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    state,
                    now,
                ),
            )
            if state == "submitted":
                connection.execute(
                    self._sql(
                        "UPDATE annotation_assignments "
                        "SET status='submitted', submitted_at=? "
                        "WHERE user_id=? AND review_type=? AND item_id=?"
                    ),
                    (now, user_id, review_type, item_id),
                )
        return revision_id

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
                "SELECT u.user_id, u.display_name, u.role, u.active, "
                "a.review_type, COUNT(a.item_id) AS assigned, "
                "SUM(CASE WHEN a.status='submitted' THEN 1 ELSE 0 END) AS submitted "
                "FROM annotation_users u "
                "LEFT JOIN annotation_assignments a ON a.user_id=u.user_id "
                "GROUP BY u.user_id, u.display_name, u.role, u.active, a.review_type "
                "ORDER BY u.user_id, a.review_type"
            ).fetchall()
        return [dict(row) for row in rows]

    def export_latest(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT revision_id, user_id, review_type, item_id, payload_json, "
                "state, created_at FROM annotation_revisions "
                "ORDER BY user_id, review_type, item_id, created_at, revision_id"
            ).fetchall()
        latest: dict[tuple[str, str, str], dict] = {}
        for row in rows:
            record = dict(row)
            record["payload"] = json.loads(record.pop("payload_json"))
            latest[(record["user_id"], record["review_type"], record["item_id"])] = record
        return list(latest.values())

    def revision_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM annotation_revisions"
            ).fetchone()
        return int(row["count"])
