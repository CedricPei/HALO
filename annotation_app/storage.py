"""Portable SQLite/PostgreSQL storage for current annotation answers."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import uuid
from collections import Counter
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

    def apply_initial_admin(
        self, migration_id: str, user_id: str, password_hash: str
    ) -> bool:
        """Create a one-time administrator on deployment without storing its password."""
        user_id = normalize_user_id(user_id)
        if user_id == self.config.admin_username.strip().lower():
            raise ValueError("This user ID belongs to the bootstrap administrator.")
        with self._connect() as connection:
            if connection.execute(
                self._sql("SELECT 1 FROM annotation_migrations WHERE migration_id=?"),
                (migration_id,),
            ).fetchone():
                return False
            if connection.execute(
                self._sql("SELECT 1 FROM annotation_users WHERE user_id=?"),
                (user_id,),
            ).fetchone():
                raise ValueError(f"{user_id} is already an annotator account.")
            connection.execute(
                self._sql(
                    "INSERT INTO annotation_admins "
                    "(user_id, password_hash, created_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(user_id) DO NOTHING"
                ),
                (user_id, password_hash, utc_now()),
            )
            connection.execute(
                self._sql(
                    "INSERT INTO annotation_migrations (migration_id, applied_at) "
                    "VALUES (?, ?) ON CONFLICT(migration_id) DO NOTHING"
                ),
                (migration_id, utc_now()),
            )
        return True

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
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT user_id, created_at, last_login_at "
                "FROM annotation_admins ORDER BY created_at, user_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def admin_exists(self, user_id: str) -> bool:
        with self._connect() as connection:
            return bool(
                connection.execute(
                    self._sql("SELECT 1 FROM annotation_admins WHERE user_id=?"),
                    (user_id,),
                ).fetchone()
            )

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
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT user_id, created_at, last_login_at "
                "FROM annotation_users ORDER BY created_at, user_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_user_codes(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT user_id, access_code FROM annotation_users "
                "ORDER BY created_at, user_id"
            ).fetchall()
        return [{"user_id": row["user_id"],
                 "access_code": "" if row["access_code"].startswith("pbkdf2_sha256$")
                 else row["access_code"]} for row in rows]

    def apply_eval_cohort(self, cohort: dict) -> bool:
        """Create the published Eval cohort atomically, once, preserving all prior work."""
        accounts = cohort["accounts"]
        ids = [normalize_user_id(account["user_id"]) for account in accounts]
        items = [item for account in accounts for item in account["item_ids"]]
        if (len(ids) != 20 or len(set(ids)) != 20 or len(items) != 4000
                or len(set(items)) != 4000
                or any(len(a["item_ids"]) != 200 for a in accounts)
                or self.config.admin_username.strip().lower() in ids):
            raise ValueError("Invalid published Eval account batch")
        for account in accounts:
            parts = account["access_code_hash"].split("$")
            if (len(parts) != 4 or parts[0] != "pbkdf2_sha256"
                    or parts[1] != str(ADMIN_PASSWORD_ITERATIONS)
                    or len(bytes.fromhex(parts[2])) != 16
                    or len(bytes.fromhex(parts[3])) != 32):
                raise ValueError("Invalid Eval access-code hash")
        with self._connect() as connection:
            claimed = connection.execute(
                self._sql("INSERT INTO annotation_migrations (migration_id, applied_at) "
                          "VALUES (?, ?) ON CONFLICT(migration_id) DO NOTHING "
                          "RETURNING migration_id"),
                (cohort["subset_id"], utc_now()),
            ).fetchone()
            if not claimed:
                return False
            now = utc_now()
            for user_id, account in zip(ids, accounts):
                exists = connection.execute(
                    self._sql("SELECT user_id FROM annotation_users WHERE user_id=? "
                              "UNION ALL SELECT user_id FROM annotation_admins WHERE user_id=?"),
                    (user_id, user_id),
                ).fetchone()
                if exists:
                    raise ValueError(f"Eval account already exists; no accounts changed: {user_id}")
                connection.execute(
                    self._sql("INSERT INTO annotation_users (user_id, access_code, created_at) "
                              "VALUES (?, ?, ?)"),
                    (user_id, account["access_code_hash"], now),
                )
                for start in range(0, len(account["item_ids"]), 100):
                    batch = account["item_ids"][start:start + 100]
                    values = ", ".join("(?, 'eval', ?, ?, 'assigned', ?)" for _ in batch)
                    connection.execute(
                        self._sql("INSERT INTO annotation_assignments "
                                  "(user_id, review_type, item_id, position, status, assigned_at) "
                                  f"VALUES {values}"),
                        tuple(value for offset, item_id in enumerate(batch)
                              for value in (user_id, item_id, start + offset, now)),
                    )
        return True

    def delete_user(self, user_id: str) -> None:
        self.delete_users([user_id])

    def rename_user_accounts(self, migration_id: str, renames: dict[str, str]) -> bool:
        """Rename participants and their work together, without changing credentials."""
        pairs = [(normalize_user_id(old), normalize_user_id(new))
                 for old, new in renames.items()]
        sources = {old for old, _ in pairs}
        targets = {new for _, new in pairs}
        if (len(sources) != len(pairs) or len(targets) != len(pairs)
                or sources & targets
                or self.config.admin_username.strip().lower() in targets):
            raise ValueError("Account renames must use distinct, unreserved target IDs")
        with self._connect() as connection:
            claimed = connection.execute(
                self._sql("INSERT INTO annotation_migrations (migration_id, applied_at) "
                          "VALUES (?, ?) ON CONFLICT(migration_id) DO NOTHING "
                          "RETURNING migration_id"),
                (migration_id, utc_now()),
            ).fetchone()
            if not claimed:
                return False
            lock = " FOR UPDATE" if not self.sqlite else ""
            for old, new in pairs:
                source = connection.execute(
                    self._sql("SELECT user_id FROM annotation_users WHERE user_id=?" + lock),
                    (old,),
                ).fetchone()
                if not source:
                    continue  # New deployments already use the short IDs; revoked users stay revoked.
                occupied = connection.execute(
                    self._sql("SELECT user_id FROM annotation_users WHERE user_id=? "
                              "UNION ALL SELECT user_id FROM annotation_admins WHERE user_id=?"),
                    (new, new),
                ).fetchone()
                if occupied:
                    raise ValueError(f"Cannot rename {old}: {new} already exists; no accounts changed")
                connection.execute(
                    self._sql("INSERT INTO annotation_users "
                              "(user_id, access_code, created_at, last_login_at) "
                              "SELECT ?, access_code, created_at, last_login_at "
                              "FROM annotation_users WHERE user_id=?"),
                    (new, old),
                )
                for table in ("annotation_assignments", "annotations"):
                    connection.execute(
                        self._sql(f"UPDATE {table} SET user_id=? WHERE user_id=?"),
                        (new, old),
                    )
                connection.execute(
                    self._sql("DELETE FROM annotation_users WHERE user_id=?"), (old,),
                )
        return True

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

    def assignment_snapshot(self, review_type: str) -> list[dict]:
        """Read assignment metadata without credentials or annotation content."""
        if review_type not in {"audit", "eval"}:
            raise ValueError("Invalid review type.")
        with self._connect() as connection:
            rows = connection.execute(
                self._sql(
                    "SELECT user_id, item_id, position, status, assigned_at, "
                    "submitted_at "
                    "FROM annotation_assignments WHERE review_type=? "
                    "ORDER BY user_id, item_id"
                ),
                (review_type,),
            ).fetchall()
        return [dict(row) for row in rows]

    def annotation_counts(self, review_type: str) -> dict[str, int]:
        """Count saved answers per account without reading their contents."""
        if review_type not in {"audit", "eval"}:
            raise ValueError("Invalid review type.")
        with self._connect() as connection:
            rows = connection.execute(
                self._sql(
                    "SELECT user_id, COUNT(*) AS count FROM annotations "
                    "WHERE review_type=? GROUP BY user_id"
                ),
                (review_type,),
            ).fetchall()
        return {row["user_id"]: int(row["count"]) for row in rows}

    def apply_fixed_audit_cohort_migration(
        self, migration_id: str, items: list[dict]
    ) -> bool:
        """Atomically restore audit001-010 to 80 tasks and remove audit011-015."""
        from annotation_app.assignment import quota_audit_assignment_plan

        user_ids = [f"audit{number:03d}" for number in range(1, 11)]
        extra_ids = [f"audit{number:03d}" for number in range(11, 16)]
        if len(items) != 400 or len({item["item_id"] for item in items}) != 400:
            raise ValueError("Audit migration requires 400 unique catalog items.")
        extra_placeholders = ", ".join("?" for _ in extra_ids)
        with self._connect() as connection:
            claimed = connection.execute(
                self._sql(
                    "INSERT INTO annotation_migrations (migration_id, applied_at) "
                    "VALUES (?, ?) ON CONFLICT(migration_id) DO NOTHING "
                    "RETURNING migration_id"
                ),
                (migration_id, utc_now()),
            ).fetchone()
            if not claimed:
                return False

            lock = " FOR UPDATE" if not self.sqlite else ""
            account_rows = connection.execute(
                "SELECT user_id FROM annotation_users WHERE user_id LIKE 'audit%' "
                f"ORDER BY user_id{lock}"
            ).fetchall()
            if {row["user_id"] for row in account_rows} != set(user_ids + extra_ids):
                raise ValueError(
                    "Audit accounts changed since diagnosis; migration was not applied."
                )
            extra_annotations = connection.execute(
                self._sql(
                    "SELECT COUNT(*) AS count FROM annotations "
                    f"WHERE user_id IN ({extra_placeholders})"
                ),
                extra_ids,
            ).fetchone()["count"]
            if extra_annotations:
                raise ValueError(
                    "One of audit011-015 has submitted an annotation; "
                    "migration was not applied."
                )
            extra_eval_assignments = connection.execute(
                self._sql(
                    "SELECT COUNT(*) AS count FROM annotation_assignments "
                    f"WHERE user_id IN ({extra_placeholders}) AND review_type!='audit'"
                ),
                extra_ids,
            ).fetchone()["count"]
            if extra_eval_assignments:
                raise ValueError(
                    "One of audit011-015 has an Eval assignment; migration was not applied."
                )

            rows = [
                dict(row)
                for row in connection.execute(
                    "SELECT user_id, item_id, position, status, assigned_at, "
                    "submitted_at FROM annotation_assignments "
                    f"WHERE review_type='audit' ORDER BY user_id, item_id{lock}"
                ).fetchall()
            ]
            if len(rows) != 800 or any(
                row["user_id"] in extra_ids and row["status"] == "submitted"
                for row in rows
            ):
                raise ValueError(
                    "Audit assignments changed since diagnosis; migration was not applied."
                )

            submitted: dict[str, set[str]] = {}
            pending: dict[str, set[str]] = {}
            for row in rows:
                target = submitted if row["status"] == "submitted" else pending
                target.setdefault(row["item_id"], set()).add(row["user_id"])
            plan = quota_audit_assignment_plan(user_ids, items, submitted, pending)
            before_annotations = [
                dict(row)
                for row in connection.execute(
                    "SELECT annotation_id, user_id, review_type, item_id, "
                    "payload_json, created_at FROM annotations "
                    "WHERE review_type='audit' ORDER BY user_id, item_id"
                ).fetchall()
            ]
            connection.execute(
                self._sql(
                    "INSERT INTO annotation_assignment_repairs "
                    "(repair_id, review_type, created_at, selected_users_json, "
                    "previous_assignments_json, saved_annotation_count) "
                    "VALUES (?, 'audit', ?, ?, ?, ?)"
                ),
                (
                    migration_id,
                    utc_now(),
                    json.dumps(user_ids),
                    json.dumps(rows, ensure_ascii=False, sort_keys=True),
                    len(before_annotations),
                ),
            )

            connection.execute(
                "DELETE FROM annotation_assignments "
                "WHERE review_type='audit' AND status='assigned'"
            )
            connection.execute(
                self._sql(
                    "DELETE FROM annotation_assignments "
                    f"WHERE user_id IN ({extra_placeholders})"
                ),
                extra_ids,
            )
            now = utc_now()
            inserts = []
            for user_id, item_ids in plan.items():
                maximum = connection.execute(
                    self._sql(
                        "SELECT COALESCE(MAX(position), -1) AS maximum "
                        "FROM annotation_assignments "
                        "WHERE review_type='audit' AND user_id=?"
                    ),
                    (user_id,),
                ).fetchone()["maximum"]
                for offset, item_id in enumerate(item_ids, start=1):
                    inserts.append((user_id, item_id, int(maximum) + offset, now))
            for start in range(0, len(inserts), 100):
                batch = inserts[start : start + 100]
                values = ", ".join(
                    "(?, 'audit', ?, ?, 'assigned', ?)" for _ in batch
                )
                connection.execute(
                    self._sql(
                        "INSERT INTO annotation_assignments "
                        "(user_id, review_type, item_id, position, status, assigned_at) "
                        f"VALUES {values}"
                    ),
                    tuple(value for row in batch for value in row),
                )
            connection.execute(
                self._sql(
                    f"DELETE FROM annotation_users WHERE user_id IN ({extra_placeholders})"
                ),
                extra_ids,
            )

            after_annotations = [
                dict(row)
                for row in connection.execute(
                    "SELECT annotation_id, user_id, review_type, item_id, "
                    "payload_json, created_at FROM annotations "
                    "WHERE review_type='audit' ORDER BY user_id, item_id"
                ).fetchall()
            ]
            active_rows = connection.execute(
                "SELECT user_id, item_id FROM annotation_assignments "
                "WHERE review_type='audit'"
            ).fetchall()
            by_user = Counter(row["user_id"] for row in active_rows)
            by_item = Counter(row["item_id"] for row in active_rows)
            catalog_ids = {item["item_id"] for item in items}
            remaining_accounts = {
                row["user_id"]
                for row in connection.execute(
                    "SELECT user_id FROM annotation_users WHERE user_id LIKE 'audit%'"
                ).fetchall()
            }
            if (
                before_annotations != after_annotations
                or len(active_rows) != 800
                or any(by_user[user_id] != 80 for user_id in user_ids)
                or set(by_user) != set(user_ids)
                or set(by_item) != catalog_ids
                or any(count != 2 for count in by_item.values())
                or remaining_accounts != set(user_ids)
            ):
                raise RuntimeError("Audit restoration failed its integrity check.")
        return True

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

    def apply_catalog_reset(
        self,
        migration_id: str,
        audit_items: list[dict],
        eval_items: list[dict],
    ) -> bool:
        """Clear answers once and rebuild assignments for the current catalogs."""
        from annotation_app.assignment import all_assignment_plan

        with self._connect() as connection:
            claimed = connection.execute(
                self._sql(
                    "INSERT INTO annotation_migrations (migration_id, applied_at) "
                    "VALUES (?, ?) ON CONFLICT(migration_id) DO NOTHING "
                    "RETURNING migration_id"
                ),
                (migration_id, utc_now()),
            ).fetchone()
            if not claimed:
                return False

            rows = connection.execute(
                "SELECT DISTINCT user_id, review_type "
                "FROM annotation_assignments ORDER BY user_id, review_type"
            ).fetchall()
            users_by_type = {
                review_type: [
                    row["user_id"]
                    for row in rows
                    if row["review_type"] == review_type
                ]
                for review_type in ("audit", "eval")
            }
            plans = {
                review_type: (
                    all_assignment_plan(
                        users_by_type[review_type], review_type, items
                    )
                    if users_by_type[review_type]
                    else {}
                )
                for review_type, items in (
                    ("audit", audit_items),
                    ("eval", eval_items),
                )
            }

            connection.execute("DELETE FROM annotations")
            connection.execute("DELETE FROM annotation_assignments")
            assigned_at = utc_now()
            assignments = []
            for review_type, plan in plans.items():
                for user_id, item_ids in plan.items():
                    for position, item_id in enumerate(item_ids):
                        assignments.append(
                            (user_id, review_type, item_id, position, assigned_at)
                        )
            for start in range(0, len(assignments), 100):
                batch = assignments[start : start + 100]
                values = ", ".join(
                    "(?, ?, ?, ?, 'assigned', ?)" for _ in batch
                )
                connection.execute(
                    self._sql(
                        "INSERT INTO annotation_assignments "
                        "(user_id, review_type, item_id, position, status, "
                        f"assigned_at) VALUES {values}"
                    ),
                    tuple(value for row in batch for value in row),
                )
        return True

    def apply_review_type_purge(
        self,
        migration_id: str,
        review_type: str,
    ) -> bool:
        """Delete one review type's answers and assignments exactly once."""
        if review_type not in {"audit", "eval"}:
            raise ValueError("Invalid review type.")
        with self._connect() as connection:
            claimed = connection.execute(
                self._sql(
                    "INSERT INTO annotation_migrations (migration_id, applied_at) "
                    "VALUES (?, ?) ON CONFLICT(migration_id) DO NOTHING "
                    "RETURNING migration_id"
                ),
                (migration_id, utc_now()),
            ).fetchone()
            if not claimed:
                return False
            connection.execute(
                self._sql("DELETE FROM annotations WHERE review_type=?"),
                (review_type,),
            )
            connection.execute(
                self._sql(
                    "DELETE FROM annotation_assignments WHERE review_type=?"
                ),
                (review_type,),
            )
        return True

    def apply_all_participant_accounts_purge(self, migration_id: str) -> bool:
        """Delete every participant account and its dependent data exactly once."""
        with self._connect() as connection:
            claimed = connection.execute(
                self._sql(
                    "INSERT INTO annotation_migrations (migration_id, applied_at) "
                    "VALUES (?, ?) ON CONFLICT(migration_id) DO NOTHING "
                    "RETURNING migration_id"
                ),
                (migration_id, utc_now()),
            ).fetchone()
            if not claimed:
                return False
            connection.execute("DELETE FROM annotations")
            connection.execute("DELETE FROM annotation_assignments")
            connection.execute("DELETE FROM annotation_users")
        return True

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
