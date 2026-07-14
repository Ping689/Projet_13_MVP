from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from langchain_mistralai import MistralAIEmbeddings

from app.config import get_settings
from app.vector_store import (
    build_event_chunks,
    build_faiss_index,
    embed_chunks,
    load_processed_events,
    save_vector_store,
)


DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "openagenda_events_processed.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "vector_store"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construit l'index vectoriel FAISS des événements OpenAgenda.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Fichier JSON des événements prétraités.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Dossier de sortie de l'index FAISS.")
    parser.add_argument("--chunk-size", type=int, default=900, help="Nombre maximal de caractères par chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Nombre de caractères de chevauchement entre chunks.")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille des lots envoyés à l'API d'embeddings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY est requis pour générer les embeddings.")

    events = load_processed_events(args.input)
    
    # Échantillonnage pour éviter de saturer l'API Mistral (maximum 200 événements par ville)
    by_city = {}
    for ev in events:
        c = (ev.get("city") or "Inconnue").lower().strip()
        by_city.setdefault(c, []).append(ev)
    
    sampled_events = []
    for city_name, city_events in by_city.items():
        sampled_events.extend(city_events[:200])
        print(f"[INFO] Ville : {city_name.capitalize()} -> {len(city_events[:200])} événements conservés sur {len(city_events)}")
        
    print(f"[INFO] Total échantillonné : {len(sampled_events)} événements sur {len(events)} initiaux.")
    events = sampled_events

    chunks = build_event_chunks(events, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    embeddings = MistralAIEmbeddings(
        api_key=settings.mistral_api_key,
        model=settings.mistral_embedding_model,
    )
    vectors = embed_chunks(chunks, embeddings, batch_size=args.batch_size)
    index = build_faiss_index(vectors)
    save_vector_store(
        index=index,
        chunks=chunks,
        output_dir=args.output_dir,
        embedding_model=settings.mistral_embedding_model,
    )
    print(f"Index FAISS enregistré avec {index.ntotal} chunks dans {args.output_dir}")


if __name__ == "__main__":
    main()
