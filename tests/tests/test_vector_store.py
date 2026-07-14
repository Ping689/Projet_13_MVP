from __future__ import annotations

import json
from pathlib import Path

import faiss

from app.vector_store import build_event_chunks, build_faiss_index, embed_chunks, save_vector_store


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "test_vector_store"


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(text.count("a") + 1), 1.0] for text in texts]


def test_build_faiss_index_from_event_chunks() -> None:
    events = [
        {
            "uid": 123,
            "title": "Atelier musique",
            "city": "Montreuil",
            "first_timing_begin": "2026-05-10T10:00:00+02:00",
            "text_for_rag": "Titre: Atelier musique. Description courte: Concert participatif.",
        }
    ]

    chunks = build_event_chunks(events, chunk_size=40, chunk_overlap=10)
    vectors = embed_chunks(chunks, FakeEmbeddings(), batch_size=2)
    index = build_faiss_index(vectors)
    save_vector_store(index=index, chunks=chunks, output_dir=TEST_OUTPUT_DIR, embedding_model="fake-embed")

    reloaded_index = faiss.read_index(str(TEST_OUTPUT_DIR / "openagenda.faiss"))
    metadata = json.loads((TEST_OUTPUT_DIR / "openagenda_metadata.json").read_text(encoding="utf-8"))

    assert reloaded_index.ntotal == len(chunks)
    assert metadata["total_chunks"] == len(chunks)
    assert metadata["chunks"][0]["event_uid"] == "123"

    (TEST_OUTPUT_DIR / "openagenda.faiss").unlink(missing_ok=True)
    (TEST_OUTPUT_DIR / "openagenda_metadata.json").unlink(missing_ok=True)
