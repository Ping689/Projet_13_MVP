from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env", override=True)


def _disable_broken_local_proxy() -> None:
    broken_proxy = "http://127.0.0.1:9"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if os.getenv(name) == broken_proxy:
            os.environ.pop(name, None)


_disable_broken_local_proxy()


@dataclass(frozen=True)
class Settings:
    mistral_api_key: str | None
    mistral_embedding_model: str
    mistral_chat_model: str
    openagenda_api_key: str | None
    openagenda_agenda_uids: str | None
    openagenda_region: str
    openagenda_city: str
    openagenda_search: str | None
    openagenda_language: str
    openagenda_allowed_cities: str | None
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str


def get_settings() -> Settings:
    return Settings(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        mistral_embedding_model=os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed"),
        mistral_chat_model=os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest"),
        openagenda_api_key=os.getenv("OPENAGENDA_API_KEY"),
        openagenda_agenda_uids=os.getenv("OPENAGENDA_AGENDA_UIDS"),
        openagenda_region=os.getenv("OPENAGENDA_REGION", "Ile-de-France"),
        openagenda_city=os.getenv("OPENAGENDA_CITY", "Paris"),
        openagenda_search=os.getenv("OPENAGENDA_SEARCH"),
        openagenda_language=os.getenv("OPENAGENDA_LANGUAGE", "fr"),
        openagenda_allowed_cities=os.getenv("OPENAGENDA_ALLOWED_CITIES"),
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=int(os.getenv("DB_PORT", "5433")),
        db_name=os.getenv("DB_NAME", "puls_events_mvp"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "password"),
    )
