from __future__ import annotations

from typing import Any

import httpx


BASE_URL = "https://api.openagenda.com/v2"


class OpenAgendaClient:
    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        if not api_key:
            raise ValueError("OPENAGENDA_API_KEY is required.")
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"key": api_key},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def search_agendas(self, search: str, size: int = 10) -> dict[str, Any]:
        response = self._client.get(
            "/agendas",
            params={"search": search, "size": size},
        )
        response.raise_for_status()
        return response.json()

    def list_events(
        self,
        agenda_uid: str,
        *,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        response = self._client.get(f"/agendas/{agenda_uid}/events", params=params)
        response.raise_for_status()
        return response.json()

    def list_public_events(
        self,
        *,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        response = self._client.get("/events", params=params)
        response.raise_for_status()
        return response.json()

    def __enter__(self) -> "OpenAgendaClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
