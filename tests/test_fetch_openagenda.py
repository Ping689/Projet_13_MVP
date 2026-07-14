from __future__ import annotations

from datetime import UTC, datetime

from scripts.fetch_openagenda import (
    build_initial_params,
    fetch_all_events,
    parse_args,
    split_agenda_uids,
)


class FakeOpenAgendaClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def list_events(self, agenda_uid: str, *, params: dict) -> dict:
        self.calls.append((agenda_uid, params))
        if len(self.calls) == 1:
            return {
                "events": [{"uid": "event-1"}],
                "after": ["cursor-1"],
            }
        return {"events": [{"uid": "event-2"}]}

    def list_public_events(self, *, params: dict) -> dict:
        raise AssertionError("fetch_openagenda.py must use agenda-specific /events endpoints.")


def test_fetch_all_events_uses_agenda_events_endpoint_and_paginates() -> None:
    client = FakeOpenAgendaClient()
    events = fetch_all_events(client, {"search": "concert", "size": 300}, "agenda-123")

    assert events == [{"uid": "event-1"}, {"uid": "event-2"}]
    assert len(client.calls) == 2
    assert client.calls[0][0] == "agenda-123"
    assert "after[]" not in client.calls[0][1]
    assert client.calls[1][1]["after[]"] == ["cursor-1"]


def test_build_initial_params_keeps_scope_filters() -> None:
    params = build_initial_params(
        region="Ile-de-France",
        city="Paris",
        search="culture",
        language="fr",
        start_at=datetime(2026, 1, 1, tzinfo=UTC),
        end_at=datetime(2026, 12, 31, tzinfo=UTC),
        size=500,
    )

    assert params["adminLevel1[]"] == ["Ile-de-France"]
    assert params["adminLevel4[]"] == ["Paris"]
    assert params["search"] == "culture"
    assert params["size"] == 300
    assert params["state[]"] == [2]


def test_agenda_uid_argument_is_supported(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["fetch_openagenda.py", "--agenda-uid", "123"])

    args = parse_args()

    assert args.agenda_uid == ["123"]


def test_split_agenda_uids_supports_comma_separated_values() -> None:
    assert split_agenda_uids("123, 456,,789") == ["123", "456", "789"]
