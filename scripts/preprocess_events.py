from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings


RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prétraite les événements OpenAgenda.")
    parser.add_argument(
        "--input",
        default="openagenda_events_raw.json",
        help="Nom du fichier d'entrée dans data/raw.",
    )
    parser.add_argument(
        "--output",
        default="openagenda_events_processed.json",
        help="Nom du fichier de sortie dans data/processed.",
    )
    return parser.parse_args()


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.replace("'", "-").replace(" ", "-").strip().lower()


def resolve_allowed_cities(configured: str | None) -> set[str]:
    if configured:
        return {
            normalize_text(city)
            for city in configured.split(",")
            if city.strip()
        }
    return set()


def filter_timings(
    timings: list[dict[str, Any]],
    start_at: datetime,
    end_at: datetime,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for timing in timings:
        begin = timing.get("begin")
        end = timing.get("end")
        if not begin or not end:
            continue
        begin_dt = datetime.fromisoformat(begin.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if end_dt >= start_at and begin_dt <= end_at:
            filtered.append(timing)
    return filtered


def build_event_text(event: dict[str, Any]) -> str:
    title = event.get("title", "")
    description = event.get("description", "")
    long_description = event.get("longDescription", "")
    location = event.get("location") or {}
    agenda = event.get("agenda") or {}
    city = location.get("city", "")
    region = location.get("region", "")
    location_name = location.get("name", "")
    address = location.get("address", "")
    keywords = ", ".join(event.get("keywords", []))

    parts = [
        f"Titre: {title}",
        f"Lieu: {location_name}",
        f"Adresse: {address}",
        f"Ville: {city}",
        f"Region: {region}",
        f"Description courte: {description}",
        f"Description longue: {long_description}",
        f"Agenda source: {agenda.get('title', '')}",
        f"Mots-cles: {keywords}",
    ]
    return "\n".join(part for part in parts if part.split(": ", 1)[1].strip())


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    location = event.get("location") or {}
    timings = event.get("timings") or []
    agenda = event.get("agenda") or {}

    return {
        "uid": event.get("uid"),
        "slug": event.get("slug"),
        "agenda_uid": agenda.get("uid") or event.get("agendaUid") or event.get("agendaUID"),
        "agenda_title": agenda.get("title"),
        "agenda_slug": agenda.get("slug"),
        "title": event.get("title"),
        "description": event.get("description"),
        "long_description": event.get("longDescription"),
        "keywords": event.get("keywords", []),
        "first_timing_begin": timings[0]["begin"] if timings else None,
        "last_timing_end": timings[-1]["end"] if timings else None,
        "timings_count": len(timings),
        "location_name": location.get("name"),
        "city": location.get("city"),
        "city_normalized": normalize_text(location.get("city")),
        "region": location.get("region"),
        "department": location.get("department"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "updated_at": event.get("updatedAt"),
        "created_at": event.get("createdAt"),
        "raw_event": event,
        "text_for_rag": build_event_text(event),
    }


def main() -> None:
    args = parse_args()
    settings = get_settings()
    input_path = RAW_DIR / args.input
    output_path = PROCESSED_DIR / args.output

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    filters = payload.get("filters", {})
    agenda_uid = str(payload.get("agenda_uid") or "")
    start_at = datetime.fromisoformat(filters["timings_gte"].replace("Z", "+00:00"))
    end_at = datetime.fromisoformat(filters["timings_lte"].replace("Z", "+00:00"))
    allowed_cities = resolve_allowed_cities(settings.openagenda_allowed_cities)

    processed_events: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        filtered_timings = filter_timings(event.get("timings") or [], start_at, end_at)
        if not filtered_timings:
            continue

        event_copy = dict(event)
        event_copy["timings"] = filtered_timings

        location = event_copy.get("location") or {}
        city_normalized = normalize_text(location.get("city"))
        if allowed_cities and city_normalized not in allowed_cities:
            continue

        processed_events.append(normalize_event(event_copy))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "source_file": str(input_path),
        "agenda_uid": agenda_uid,
        "agenda_uids": payload.get("agenda_uids", []),
        "collection_mode": payload.get("collection_mode", "agenda"),
        "filters": filters,
        "total_events": len(processed_events),
        "events": processed_events,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(processed_events)} événements prétraités enregistrés dans {output_path}")


if __name__ == "__main__":
    main()
