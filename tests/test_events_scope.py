from __future__ import annotations

import json
import os
import unicodedata
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "openagenda_events_processed.json"


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.replace("'", "-").replace(" ", "-").strip().lower()


def load_payload() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def load_events() -> list[dict]:
    return load_payload()["events"]


def parse_allowed_cities(value: str | None) -> set[str]:
    if not value:
        return set()
    return {normalize_text(city) for city in value.split(",") if city.strip()}


def test_processed_dataset_exists() -> None:
    assert DATASET_PATH.exists(), "Run preprocess_events.py before running tests."


def test_events_match_the_selected_geographic_scope() -> None:
    payload = load_payload()
    events = load_events()
    assert events, "Processed dataset is empty."
    filters = payload.get("filters", {})
    allowed_cities = parse_allowed_cities(os.getenv("OPENAGENDA_ALLOWED_CITIES"))
    expected_city = normalize_text(filters.get("city"))
    expected_region = normalize_text(filters.get("region"))

    for event in events:
        city_normalized = normalize_text(event.get("city"))
        if allowed_cities:
            if not city_normalized:
                continue
            assert city_normalized in allowed_cities, (
                f"Event {event['uid']} is outside the selected city scope: {event.get('city')}"
            )
        elif expected_city:
            assert city_normalized == expected_city, (
                f"Event {event['uid']} is outside the selected city scope: {event.get('city')}"
            )
        else:
            region = normalize_text(event.get("region"))
            assert region == expected_region, (
                f"Event {event['uid']} is outside the selected region scope: {event.get('region')}"
            )


def test_events_timings_are_within_the_selected_time_window() -> None:
    payload = load_payload()
    events = load_events()
    filters = payload["filters"]
    lower_bound = datetime.fromisoformat(filters["timings_gte"].replace("Z", "+00:00"))
    upper_bound = datetime.fromisoformat(filters["timings_lte"].replace("Z", "+00:00"))

    for event in events:
        first_timing_begin = event.get("first_timing_begin")
        last_timing_end = event.get("last_timing_end")
        assert first_timing_begin, f"Event {event['uid']} has no timing."
        assert last_timing_end, f"Event {event['uid']} has no timing end."
        first_dt = datetime.fromisoformat(first_timing_begin.replace("Z", "+00:00"))
        last_dt = datetime.fromisoformat(last_timing_end.replace("Z", "+00:00"))
        assert last_dt >= lower_bound and first_dt <= upper_bound, (
            f"Event {event['uid']} is outside the selected time window."
        )
