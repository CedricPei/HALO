"""Portable SQLite/PostgreSQL storage for current annotation answers."""

from __future__ import annotations

import atexit
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
ADMIN_PASSWORD_ITERATIONS = 600_000


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


def generate_admin_password() -> str:
    return secrets.token_urlsafe(32)


def hash_admin_password(password: str) -> str:
    if len(password) < 20:
        raise ValueError("Administrator passwords must have at least 20 characters.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, ADMIN_PASSWORD_ITERATIONS
    )
    return f"pbkdf2_sha256${ADMIN_PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_admin_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = password_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError):
        return False


class AnnotationStore:
    """Central review store; original benchmark files are never mutated."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.sqlite = config.database_url.startswith("sqlite:///")
        self._pool = None
        if self.sqlite:
            self.sqlite_path = Path(config.database_url.removeprefix("sqlite:///"))
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            try:
                from psycopg.rows import dict_row
                from psycopg_pool import ConnectionPool
            except ImportError as error:
                raise RuntimeError(
                    "PostgreSQL mode requires psycopg and psycopg-pool."
                ) from error
            self._pool = ConnectionPool(
                self.config.database_url,
                kwargs={
                    "row_factory": dict_row,
                    "prepare_threshold": None,
                    "connect_timeout": 10,
                },
                min_size=1,
                max_size=8,
                timeout=15,
                check=ConnectionPool.check_connection,
                open=True,
            )
        try:
            self.ensure_schema()
        except Exception:
            self.close()
            raise
        if self._pool is not None:
            atexit.register(self.close)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            atexit.unregister(self.close)

    @contextmanager
    def _connect(self):
        if not self.sqlite:
            # Each checkout owns its transaction; sessions never share one.
            with self._pool.connection() as connection:
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

    def _fetch_all(self, statement: str, parameters: tuple = ()) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(self._sql(statement), parameters).fetchall()
        return [dict(row) for row in rows]

    def _fetch_one(self, statement: str, parameters: tuple = ()) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(self._sql(statement), parameters).fetchone()
        return dict(row) if row is not None else None

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

    def create_user(
        self,
        user_id: str,
        access_code: str,
    ) -> None:
        user_id = normalize_user_id(user_id)
        if user_id == self.config.admin_username.strip().lower():
            raise ValueError("This user ID is reserved for the bootstrap administrator.")
        with self._connect() as connection:
            if connection.execute(
                self._sql("SELECT 1 FROM annotation_admins WHERE user_id=?"),
                (user_id,),
            ).fetchone():
                raise ValueError("This user ID belongs to an administrator.")
            connection.execute(
                self._sql(
                    "INSERT INTO annotation_users "
                    "(user_id, access_code, created_at) VALUES (?, ?, ?)"
                ),
                (
                    user_id,
                    access_code.strip().upper(),
                    utc_now(),
                ),
            )

    def create_admin(self, user_id: str, password: str) -> None:
        user_id = normalize_user_id(user_id)
        password_hash = hash_admin_password(password)
        if user_id == self.config.admin_username.strip().lower():
            raise ValueError("This user ID belongs to the bootstrap administrator.")
        with self._connect() as connection:
            if connection.execute(
                self._sql("SELECT 1 FROM annotation_users WHERE user_id=?"),
                (user_id,),
            ).fetchone():
                raise ValueError("This user ID belongs to an annotator.")
            connection.execute(
                self._sql(
                    "INSERT INTO annotation_admins "
                    "(user_id, password_hash, created_at) VALUES (?, ?, ?)"
                ),
                (user_id, password_hash, utc_now()),
            )

    def authenticate_admin(self, user_id: str, password: str) -> dict | None:
        try:
            user_id = normalize_user_id(user_id)
        except ValueError:
            return None
        with self._connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT user_id, password_hash FROM annotation_admins WHERE user_id=?"
                ),
                (user_id,),
            ).fetchone()
            if not row or not verify_admin_password(password, row["password_hash"]):
                return None
            connection.execute(
                self._sql(
                    "UPDATE annotation_admins SET last_login_at=? WHERE user_id=?"
                ),
                (utc_now(), user_id),
            )
        return {"user_id": user_id, "is_admin": True, "admin_source": "database"}

    def list_admins(self) -> list[dict]:
        return self._fetch_all(
            "SELECT user_id, created_at, last_login_at "
            "FROM annotation_admins ORDER BY created_at, user_id"
        )

    def admin_exists(self, user_id: str) -> bool:
        return self._fetch_one(
            "SELECT 1 FROM annotation_admins WHERE user_id=?", (user_id,)
        ) is not None

    def delete_admin(self, user_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                self._sql("DELETE FROM annotation_admins WHERE user_id=?"),
                (normalize_user_id(user_id),),
            )

    def authenticate(self, user_id: str, access_code: str) -> dict | None:
        try:
            user_id = normalize_user_id(user_id)
        except ValueError:
            return None
        with self._connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT user_id, access_code FROM annotation_users "
                    "WHERE user_id=?"
                ),
                (user_id,),
            ).fetchone()
            if not row:
                return None
            supplied = access_code.strip().upper()
            stored = row["access_code"]
            valid = (verify_admin_password(supplied, stored)
                     if stored.startswith("pbkdf2_sha256$")
                     else hmac.compare_digest(stored.encode(), supplied.encode()))
            if not valid:
                return None
            connection.execute(
                self._sql(
                    "UPDATE annotation_users SET last_login_at=? WHERE user_id=?"
                ),
                (utc_now(), user_id),
            )
        return {"user_id": row["user_id"]}

    def list_users(self) -> list[dict]:
        return self._fetch_all(
            "SELECT user_id, created_at, last_login_at "
            "FROM annotation_users ORDER BY created_at, user_id"
        )

    def list_user_codes(self) -> list[dict]:
        rows = self._fetch_all(
            "SELECT user_id, access_code FROM annotation_users "
            "ORDER BY created_at, user_id"
        )
        return [{"user_id": row["user_id"],
                 "access_code": "" if row["access_code"].startswith("pbkdf2_sha256$")
                 else row["access_code"]} for row in rows]

    def delete_user(self, user_id: str) -> None:
        self.delete_users([user_id])

    def delete_users(self, user_ids: Iterable[str]) -> None:
        user_ids = list(dict.fromkeys(normalize_user_id(value) for value in user_ids))
        if not user_ids:
            return
        placeholders = ", ".join("?" for _ in user_ids)
        with self._connect() as connection:
            for table in (
                "annotations",
                "annotation_assignments",
                "annotation_users",
            ):
                connection.execute(
                    self._sql(
                        f"DELETE FROM {table} WHERE user_id IN ({placeholders})"
                    ),
                    user_ids,
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
            self._insert_assignments(connection, user_id, review_type, item_ids, now)

    def _insert_assignments(
        self, connection, user_id: str, review_type: str,
        item_ids: Iterable[str], assigned_at: str, start: int = 0,
    ) -> None:
        statement = self._sql(
            "INSERT INTO annotation_assignments "
            "(user_id, review_type, item_id, position, status, assigned_at) "
            "VALUES (?, ?, ?, ?, 'assigned', ?)"
        )
        for position, item_id in enumerate(item_ids, start=start):
            connection.execute(
                statement, (user_id, review_type, item_id, position, assigned_at)
            )

    def assignment_ids(self, user_id: str, review_type: str) -> list[str]:
        rows = self._fetch_all(
            "SELECT item_id FROM annotation_assignments "
            "WHERE user_id=? AND review_type=? ORDER BY position, item_id",
            (user_id, review_type),
        )
        return [row["item_id"] for row in rows]

    def review_state(self, user_id: str) -> dict | None:
        """Read a participant's current assignments and answers in one query.

        Do not cache this result: revoked accounts, reassigned work, and saved
        reviews must be visible on the next interaction.
        """
        rows = self._fetch_all(
            "SELECT a.review_type, a.item_id, a.status, "
            "n.annotation_id, n.payload_json, n.created_at "
            "FROM annotation_users u "
            "LEFT JOIN annotation_assignments a ON a.user_id=u.user_id "
            "LEFT JOIN annotations n ON n.user_id=a.user_id "
            "AND n.review_type=a.review_type AND n.item_id=a.item_id "
            "WHERE u.user_id=? ORDER BY a.review_type, a.position, a.item_id",
            (user_id,),
        )
        if not rows:
            return None
        state = {
            name: {"item_ids": [], "reviewed_ids": set(), "annotations": {}}
            for name in ("audit", "eval")
        }
        for row in rows:
            if row["review_type"] is None:
                continue
            review = state[row["review_type"]]
            item_id = row["item_id"]
            review["item_ids"].append(item_id)
            if row["status"] == "submitted":
                review["reviewed_ids"].add(item_id)
            if row["annotation_id"] is not None:
                review["annotations"][item_id] = {
                    "annotation_id": row["annotation_id"],
                    "payload": json.loads(row["payload_json"]),
                    "created_at": row["created_at"],
                }
        return state

    def assignment_snapshot(self, review_type: str) -> list[dict]:
        """Read assignment metadata without credentials or annotation content."""
        if review_type not in {"audit", "eval"}:
            raise ValueError("Invalid review type.")
        return self._fetch_all(
            "SELECT user_id, item_id, position, status, assigned_at, submitted_at "
            "FROM annotation_assignments WHERE review_type=? ORDER BY user_id, item_id",
            (review_type,),
        )

    def annotation_counts(self, review_type: str) -> dict[str, int]:
        """Count saved answers per account without reading their contents."""
        if review_type not in {"audit", "eval"}:
            raise ValueError("Invalid review type.")
        rows = self._fetch_all(
            "SELECT user_id, COUNT(*) AS count FROM annotations "
            "WHERE review_type=? GROUP BY user_id",
            (review_type,),
        )
        return {row["user_id"]: int(row["count"]) for row in rows}

    def reviewed_ids(self, user_id: str, review_type: str) -> set[str]:
        rows = self._fetch_all(
            "SELECT item_id FROM annotation_assignments "
            "WHERE user_id=? AND review_type=? AND status='submitted'",
            (user_id, review_type),
        )
        return {row["item_id"] for row in rows}

    def submitted_users_by_item(self, review_type: str) -> dict[str, set[str]]:
        rows = self._fetch_all(
            "SELECT item_id, user_id FROM annotation_assignments "
            "WHERE review_type=? AND status='submitted'",
            (review_type,),
        )
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
                self._insert_assignments(
                    connection, user_id, review_type, item_ids, now, start=start
                )

    def get_annotation(
        self, user_id: str, review_type: str, item_id: str
    ) -> dict | None:
        row = self._fetch_one(
            "SELECT annotation_id, payload_json, created_at FROM annotations "
            "WHERE user_id=? AND review_type=? AND item_id=? LIMIT 1",
            (user_id, review_type, item_id),
        )
        if row is None:
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
            lock = " FOR UPDATE" if not self.sqlite else ""
            assigned = connection.execute(
                self._sql(
                    "SELECT 1 FROM annotation_assignments "
                    f"WHERE user_id=? AND review_type=? AND item_id=?{lock}"
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
            updated = connection.execute(
                self._sql(
                    "UPDATE annotation_assignments "
                    "SET status='submitted', submitted_at=? "
                    "WHERE user_id=? AND review_type=? AND item_id=?"
                ),
                (now, user_id, review_type, item_id),
            )
            if updated.rowcount != 1:
                raise ValueError("This assignment changed while the review was saved.")
        return annotation_id

    def progress_for_user(self, user_id: str) -> dict[str, dict[str, int]]:
        progress = {
            "audit": {"assigned": 0, "submitted": 0},
            "eval": {"assigned": 0, "submitted": 0},
        }
        rows = self._fetch_all(
            "SELECT review_type, COUNT(*) AS assigned, "
            "SUM(CASE WHEN status='submitted' THEN 1 ELSE 0 END) AS submitted "
            "FROM annotation_assignments WHERE user_id=? GROUP BY review_type",
            (user_id,),
        )
        for row in rows:
            progress[row["review_type"]] = {
                "assigned": int(row["assigned"]),
                "submitted": int(row["submitted"] or 0),
            }
        return progress

    def progress_all_users(self) -> list[dict]:
        return self._fetch_all(
            "SELECT u.user_id, a.review_type, COUNT(a.item_id) AS assigned, "
            "SUM(CASE WHEN a.status='submitted' THEN 1 ELSE 0 END) AS submitted "
            "FROM annotation_users u "
            "LEFT JOIN annotation_assignments a ON a.user_id=u.user_id "
            "GROUP BY u.user_id, a.review_type ORDER BY u.user_id, a.review_type"
        )

    def export_annotations(self) -> list[dict]:
        rows = self._fetch_all(
            "SELECT annotation_id, user_id, review_type, item_id, payload_json, "
            "created_at FROM annotations ORDER BY user_id, review_type, item_id"
        )
        for row in rows:
            row["payload"] = json.loads(row.pop("payload_json"))
        return rows
