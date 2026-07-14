from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.openagenda_client import OpenAgendaClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recherche des agendas OpenAgenda.")
    parser.add_argument(
        "--search",
        default="Paris culture",
        help="Texte de recherche utilisé sur les agendas OpenAgenda.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10,
        help="Nombre maximal d'agendas à afficher.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    with OpenAgendaClient(settings.openagenda_api_key or "") as client:
        payload = client.search_agendas(search=args.search, size=args.size)

    agendas = payload.get("agendas", [])
    if not agendas:
        print("Aucun agenda trouvé.")
        return

    for agenda in agendas:
        title = agenda.get("title")
        uid = agenda.get("uid")
        description = agenda.get("description", "")
        print(f"UID: {uid}")
        print(f"Titre: {title}")
        if description:
            print(f"Description: {description[:180]}")
        print("-" * 60)


if __name__ == "__main__":
    main()