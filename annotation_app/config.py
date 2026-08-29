"""Configuration loaded from environment variables or Streamlit Secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        import streamlit as st

        return str(st.secrets.get(name, default))
    except (FileNotFoundError, KeyError, RuntimeError):
        return default


@dataclass(frozen=True)
class AppConfig:
    database_url: str
    sqlite_path: Path
    admin_username: str
    admin_password: str
    auth_pepper: str

    @property
    def backend_name(self) -> str:
        return "Supabase PostgreSQL" if self.database_url else "Local SQLite preview"

    @property
    def is_cloud(self) -> bool:
        return bool(self.database_url)

    @classmethod
    def load(cls) -> "AppConfig":
        sqlite_value = _secret(
            "ANNOTATION_DB_PATH",
            str(APP_DIR / "local" / "annotations.sqlite"),
        )
        return cls(
            database_url=_secret("DATABASE_URL").strip(),
            sqlite_path=Path(sqlite_value).expanduser(),
            admin_username=_secret("ADMIN_USERNAME", "admin").strip() or "admin",
            admin_password=_secret("ADMIN_PASSWORD"),
            auth_pepper=_secret("AUTH_PEPPER"),
        )

    def missing_secrets(self) -> list[str]:
        missing = []
        if not self.admin_password:
            missing.append("ADMIN_PASSWORD")
        if not self.auth_pepper:
            missing.append("AUTH_PEPPER")
        return missing
