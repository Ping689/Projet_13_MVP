from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import RunnableLambda

from app.rag_chain import answer_question, format_context
from app.vector_store import SearchResult, build_event_chunks, build_faiss_index, embed_chunks, save_vector_store


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "test_rag_store"


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float("concert" in text.lower()), float("enfant" in text.lower()), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float("concert" in text.lower()), float("enfant" in text.lower()), 1.0]


def test_format_context_includes_source_metadata() -> None:
    chunks = build_event_chunks(
        [
            {
                "uid": 1,
                "title": "Concert jazz",
                "city": "Montreuil",
                "first_timing_begin": "2026-05-10T20:00:00+02:00",
                "text_for_rag": "Titre: Concert jazz Ville: Montreuil",
            }
        ]
    )
    vectors = embed_chunks(chunks, FakeEmbeddings())
    index = build_faiss_index(vectors)
    index_results = index.search(vectors[:1], 1)
    context = format_context([SearchResult(chunk=chunks[0], score=0.99)])

    assert index_results[1][0][0] == 0
    assert "Concert jazz" in context
    assert "Montreuil" in context


def test_answer_question_uses_retrieval_and_langchain_chain() -> None:
    events = [
        {
            "uid": 1,
            "title": "Concert jazz",
            "city": "Montreuil",
            "first_timing_begin": "2026-05-10T20:00:00+02:00",
            "text_for_rag": "Titre: Concert jazz Description courte: Concert pour tous. Ville: Montreuil",
        }
    ]
    chunks = build_event_chunks(events)
    vectors = embed_chunks(chunks, FakeEmbeddings())
    index = build_faiss_index(vectors)
    save_vector_store(index=index, chunks=chunks, output_dir=TEST_OUTPUT_DIR, embedding_model="fake-embed")

    fake_chat = RunnableLambda(lambda _prompt: "Je recommande Concert jazz a Montreuil.")
    result = answer_question(
        question="Quel concert recommander ?",
        embeddings=FakeEmbeddings(),
        chat_model=fake_chat,
        vector_store_dir=TEST_OUTPUT_DIR,
        top_k=1,
    )
    context = format_context(result.sources)

    assert "Concert jazz" in result.answer
    assert "Montreuil" in context
    assert result.sources[0].chunk.event_uid == "1"

    (TEST_OUTPUT_DIR / "openagenda.faiss").unlink(missing_ok=True)
    (TEST_OUTPUT_DIR / "openagenda_metadata.json").unlink(missing_ok=True)


def test_answer_question_filters_sources_by_allowed_city() -> None:
    events = [
        {
            "uid": 1,
            "title": "Festival enfant",
            "city": "Montreuil",
            "first_timing_begin": "2026-05-10T10:00:00+02:00",
            "text_for_rag": "Titre: Festival enfant Ville: Montreuil",
        },
        {
            "uid": 2,
            "title": "Atelier enfant Paris",
            "city": "Paris",
            "first_timing_begin": "2026-05-11T10:00:00+02:00",
            "text_for_rag": "Titre: Atelier enfant Paris Ville: Paris",
        },
    ]
    chunks = build_event_chunks(events)
    vectors = embed_chunks(chunks, FakeEmbeddings())
    index = build_faiss_index(vectors)
    save_vector_store(index=index, chunks=chunks, output_dir=TEST_OUTPUT_DIR, embedding_model="fake-embed")

    fake_chat = RunnableLambda(lambda _prompt: "Je recommande Atelier enfant Paris.")
    result = answer_question(
        question="Quel evenement enfant recommander ?",
        embeddings=FakeEmbeddings(),
        chat_model=fake_chat,
        vector_store_dir=TEST_OUTPUT_DIR,
        top_k=2,
        allowed_cities={"Paris"},
    )

    assert result.sources
    assert {source.chunk.city for source in result.sources} == {"Paris"}

    (TEST_OUTPUT_DIR / "openagenda.faiss").unlink(missing_ok=True)
    (TEST_OUTPUT_DIR / "openagenda_metadata.json").unlink(missing_ok=True)
