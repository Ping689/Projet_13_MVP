"""Récupère les événements OpenAgenda du périmètre choisi et sauvegarde le JSON brut."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.openagenda_client import OpenAgendaClient


RAW_DIR = ROOT_DIR / "data" / "raw"


def isoformat_z(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Récupère des événements depuis OpenAgenda.")
    parser.add_argument(
        "--agenda-uid",
        action="append",
        default=None,
        help="UID d'agenda OpenAgenda. Peut etre repete. Par defaut, utilise OPENAGENDA_AGENDA_UIDS.",
    )
    parser.add_argument(
        "--search",
        action="append",
        default=None,
        help=(
            "Recherche texte pour découvrir les événements. Peut être répétée. "
            "Par défaut, utilise OPENAGENDA_SEARCH depuis .env."
        ),
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Filtre de région, par exemple Ile-de-France.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Filtre de ville, par exemple Paris.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=365,
        help="Nombre de jours passés à inclure.",
    )
    parser.add_argument(
        "--days-forward",
        type=int,
        default=365,
        help="Nombre de jours futurs à inclure.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=300,
        help="Taille de lot par appel API. Le maximum OpenAgenda est 300.",
    )
    parser.add_argument(
        "--output",
        default="openagenda_events_raw.json",
        help="Nom du fichier de sortie écrit dans data/raw.",
    )
    return parser.parse_args()


def build_initial_params(
    *,
    region: str | None,
    city: str | None,
    search: str | None,
    language: str,
    start_at: datetime,
    end_at: datetime,
    size: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "detailed": 1,
        "monolingual": language,
        "size": min(size, 300),
        "relative[]": ["passed", "current", "upcoming"],
        "timings[gte]": isoformat_z(start_at),
        "timings[lte]": isoformat_z(end_at),
        "state[]": [2],
    }
    if region:
        params["adminLevel1[]"] = [region]
    if city:
        params["adminLevel4[]"] = [city]
    if search:
        params["search"] = search
    return params


def fetch_all_events(
    client: OpenAgendaClient,
    params: dict[str, Any],
    agenda_uid: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    after: list[str] | None = None

    while True:
        current_params = dict(params)
        if after:
            current_params["after[]"] = after

        payload = client.list_events(agenda_uid, params=current_params)
        batch = payload.get("events", [])
        events.extend(batch)
        after = payload.get("after")

        if not after:
            break

    return events


def split_searches(configured: str | None) -> list[str]:
    if not configured:
        return []
    return [value.strip() for value in configured.split(",") if value.strip()]


def split_agenda_uids(configured: str | None) -> list[str]:
    if not configured:
        return []
    return [value.strip() for value in configured.split(",") if value.strip()]


def event_dedupe_key(event: dict[str, Any]) -> str:
    agenda = event.get("agenda") or {}
    agenda_uid = agenda.get("uid") or event.get("agendaUid") or event.get("agendaUID") or ""
    uid = event.get("uid") or ""
    slug = event.get("slug") or ""
    return f"{agenda_uid}:{uid}:{slug}"


def main() -> None:
    args = parse_args()
    settings = get_settings()

    agenda_uids = args.agenda_uid or split_agenda_uids(settings.openagenda_agenda_uids)
    if not agenda_uids:
        raise ValueError("OPENAGENDA_AGENDA_UIDS ou --agenda-uid est requis pour recuperer les evenements.")

    region = args.region or settings.openagenda_region
    city = args.city or settings.openagenda_city
    searches = args.search if args.search is not None else split_searches(settings.openagenda_search)
    now = datetime.now(UTC)
    start_at = now - timedelta(days=args.days_back)
    end_at = now + timedelta(days=args.days_forward)

    with OpenAgendaClient(settings.openagenda_api_key or "") as client:
        events_by_key: dict[str, dict[str, Any]] = {}
        search_scope = searches or [None]
        for agenda_uid in agenda_uids:
            for search in search_scope:
                params = build_initial_params(
                    region=region,
                    city=city,
                    search=search,
                    language=settings.openagenda_language,
                    start_at=start_at,
                    end_at=end_at,
                    size=args.size,
                )
                for event in fetch_all_events(client, params, agenda_uid):
                    events_by_key[event_dedupe_key(event)] = event
        events = list(events_by_key.values())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / args.output
    payload = {
        "fetched_at": isoformat_z(now),
        "agenda_uid": agenda_uids[0] if len(agenda_uids) == 1 else None,
        "agenda_uids": agenda_uids,
        "collection_mode": "agendas",
        "filters": {
            "region": region,
            "city": city,
            "searches": searches,
            "timings_gte": isoformat_z(start_at),
            "timings_lte": isoformat_z(end_at),
            "language": settings.openagenda_language,
        },
        "total_events": len(events),
        "events": events,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(events)} événements enregistrés dans {output_path}")


if __name__ == "__main__":
    main()
