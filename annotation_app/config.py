"""Configuration loaded from environment variables or Streamlit Secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass


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
    admin_username: str
    admin_password: str
    auth_pepper: str

    @classmethod
    def load(cls) -> "AppConfig":
        return cls(
            database_url=_secret("DATABASE_URL").strip(),
            admin_username=_secret("ADMIN_USERNAME", "admin").strip() or "admin",
            admin_password=_secret("ADMIN_PASSWORD"),
            auth_pepper=_secret("AUTH_PEPPER"),
        )

    def missing_secrets(self) -> list[str]:
        values = {
            "DATABASE_URL": self.database_url,
            "ADMIN_PASSWORD": self.admin_password,
            "AUTH_PEPPER": self.auth_pepper,
        }
        return [name for name, value in values.items() if not value]
