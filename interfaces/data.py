"""Load review items and persist Audit/Eval decisions."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from evaluation.intensity import INTENSITY_LEVELS


ROOT_DIR = Path(__file__).resolve().parents[1]
DOMAINS_DIR = ROOT_DIR / "domains"
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"
REVIEW_PATH = ROOT_DIR / "annotations" / "reviews.sqlite"
REVIEWER = "default"
REVIEW_TABLES = {
    "audit": "construction_reviews",
    "eval": "evaluation_reviews",
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _state_summary(instance_dir, instance_id):
    data_dir = instance_dir / "data"
    database_path = data_dir / f"{instance_id}.sqlite"
    database = {}
    if database_path.exists():
        connection = sqlite3.connect(database_path)
        try:
            tables = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            for table in tables:
                columns = [
                    row[1]
                    for row in connection.execute(f'PRAGMA table_info("{table}")')
                ]
                rows = connection.execute(f'SELECT * FROM "{table}" LIMIT 50').fetchall()
                database[table] = {"columns": columns, "rows": rows}
        finally:
            connection.close()

    files = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(data_dir.glob("*.txt"))
    }
    apps = {
        path.stem: _json(path)
        for path in sorted((instance_dir / "app").glob("*.json"))
    }
    return {"database": database, "files": files, "apps": apps}


def load_construction_items():
    """Return every accepted Scenario with enough context for human audit."""
    items = []
    state_cache = {}
    for public_path in sorted(DOMAINS_DIR.glob("*/instances/*/scenarios/*/*.json")):
        instance_dir = public_path.parents[2]
        instance_id = instance_dir.name
        domain_dir = instance_dir.parents[1]
        trigger = public_path.parent.name
        scenario_id = public_path.stem
        hidden_path = (
            instance_dir / "scenarios" / "logs" / scenario_id
            / "scenario_output.json"
        )
        if instance_dir not in state_cache:
            state_cache[instance_dir] = _state_summary(instance_dir, instance_id)
        items.append(
            {
                "item_id": f"{domain_dir.name}/{instance_id}/{scenario_id}",
                "domain_id": domain_dir.name,
                "instance_id": instance_id,
                "scenario_id": scenario_id,
                "trigger_type": trigger,
                "instance": _json(instance_dir / "profile.json"),
                "scenario": _json(hidden_path),
                "state": state_cache[instance_dir],
                "path": str(public_path.relative_to(ROOT_DIR)),
            }
        )
    return items


def load_evaluation_items():
    """Flatten saved evaluation runs into checkpoint-level review items."""
    items = []
    if not RESULTS_DIR.exists():
        return items
    for path in sorted(RESULTS_DIR.glob("*.json")):
        run = _json(path)
        scenario = _json(ROOT_DIR / run["source_path"])
        for checkpoint in run["checkpoints"]:
            if checkpoint["agent"]["decision"] == "error":
                continue
            event_id = checkpoint["event_id"]
            items.append(
                {
                    "item_id": f"{run['run_id']}/{event_id}",
                    "model": run["model"],
                    "domain_id": run["domain_id"],
                    "trigger_type": run["trigger_type"],
                    "scenario_description": scenario["scenario_description"],
                    "event_prefix": checkpoint["event_prefix"],
                    "agent": checkpoint["agent"],
                }
            )
    return items


TIMING_ERRORS = ("Premature", "Belated", "Inaction")
INTENSITY_ERRORS = ("Insufficient", "Excessive")
OTHER_ISSUES = (
    "Unsupported Evidence",
    "Irrelevant Action",
    "Factual Contradiction",
    "Others",
)


class ReviewStore:
    """SQLite-backed, idempotent storage for human decisions."""

    def __init__(self):
        self.path = REVIEW_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS construction_reviews (
                    reviewer TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    early_event TEXT,
                    late_event TEXT,
                    ratings_json TEXT NOT NULL,
                    issues TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (reviewer, item_id)
                );
                CREATE TABLE IF NOT EXISTS evaluation_reviews (
                    reviewer TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    acceptable TEXT NOT NULL,
                    primary_reason TEXT,
                    timing TEXT,
                    intensity TEXT,
                    diagnostics_json TEXT NOT NULL,
                    comments TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (reviewer, item_id)
                );
                """
            )

    def save_audit(self, item_id, verdict, comment):
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO construction_reviews (
                    reviewer, item_id, verdict, early_event, late_event,
                    ratings_json, issues, revision, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(reviewer, item_id) DO UPDATE SET
                    verdict=excluded.verdict,
                    early_event=excluded.early_event,
                    late_event=excluded.late_event,
                    ratings_json=excluded.ratings_json,
                    issues=excluded.issues,
                    revision=excluded.revision,
                    updated_at=excluded.updated_at
                """,
                (
                    REVIEWER,
                    item_id,
                    verdict,
                    "",
                    "",
                    "{}",
                    comment,
                    "",
                    now,
                    now,
                ),
            )

    def save_eval(self, item_id, decision):
        acceptable = decision["acceptable"]
        primary_reason = decision["primary_reason"]
        timing = decision["timing"]
        intensity = decision["intensity"]
        diagnostics = decision["diagnostics"]
        comments = decision["comments"]
        if acceptable not in {"acceptable", "unacceptable"}:
            raise ValueError("acceptable must be acceptable or unacceptable")
        if diagnostics.get("action_intensity") not in INTENSITY_LEVELS:
            raise ValueError("action_intensity must be one of L0, L1, L2, or L3")
        if acceptable == "acceptable":
            primary_reason, timing, intensity = "", "Timely", "Appropriate"
        elif primary_reason == "Timing" and timing not in TIMING_ERRORS:
            raise ValueError("a Timing rejection requires a timing error")
        elif primary_reason == "Intensity" and intensity not in INTENSITY_ERRORS:
            raise ValueError("an Intensity rejection requires an intensity error")
        elif primary_reason == "Other issue":
            other_issue = diagnostics.get("other_issue")
            if other_issue not in OTHER_ISSUES:
                raise ValueError("an Other issue rejection requires a listed issue")
            if other_issue == "Others" and not str(
                diagnostics.get("other_issue_detail", "")
            ).strip():
                raise ValueError("Others requires a description")
        elif primary_reason not in {"Timing", "Intensity", "Other issue"}:
            raise ValueError("an unacceptable response requires a rejection area")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(reviewer, item_id) DO UPDATE SET
                    acceptable=excluded.acceptable,
                    primary_reason=excluded.primary_reason,
                    timing=excluded.timing,
                    intensity=excluded.intensity,
                    diagnostics_json=excluded.diagnostics_json,
                    comments=excluded.comments,
                    updated_at=excluded.updated_at
                """,
                (
                    REVIEWER,
                    item_id,
                    acceptable,
                    primary_reason,
                    timing,
                    intensity,
                    json.dumps(diagnostics, ensure_ascii=False),
                    comments,
                    now,
                    now,
                ),
            )

    def get(self, review_type, item_id):
        table = REVIEW_TABLES[review_type]
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {table} WHERE reviewer=? AND item_id=?",
                (REVIEWER, item_id),
            ).fetchone()
        return dict(row) if row else None

    def reviewed_ids(self, review_type):
        table = REVIEW_TABLES[review_type]
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT item_id FROM {table} WHERE reviewer=?", (REVIEWER,)
            ).fetchall()
        return {row["item_id"] for row in rows}
