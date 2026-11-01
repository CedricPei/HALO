"""SQLite schema inspection and SQL execution."""

import sqlite3
from pathlib import Path


class Database:
    """Keep one SQLite connection for the lifetime of the object."""

    def __init__(self, database_path):
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.execute("PRAGMA foreign_keys = ON")

    def get_schema(self):
        """Return user-created tables and basic column metadata."""
        tables = self.connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return {
            "tables": [
                {
                    "name": name,
                    "columns": [
                        {
                            "name": row[1],
                            "type": row[2],
                            "not_null": bool(row[3]),
                            "default_value": row[4],
                            "primary_key": bool(row[5]),
                        }
                        for row in self.connection.execute(
                            f'PRAGMA table_info("{name}")'
                        ).fetchall()
                    ],
                }
                for (name,) in tables
            ]
        }

    def execute(self, sql, parameters=()):
        """Execute one SQL statement and return rows, an affected count, or an error."""
        try:
            cursor = self.connection.execute(sql, parameters)
            if cursor.description:
                return {
                    "columns": [column[0] for column in cursor.description],
                    "rows": [list(row) for row in cursor.fetchall()],
                }
            self.connection.commit()
            return {"affected_rows": max(cursor.rowcount, 0)}
        except sqlite3.Error as error:
            self.connection.rollback()
            return {"error": str(error)}
