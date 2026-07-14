from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag_chain import DEFAULT_VECTOR_STORE_DIR, answer_question


app = FastAPI(
    title="Puls-Events RAG API",
    description="API de test Q/R pour le chatbot RAG OpenAgenda.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceResponse(BaseModel):
    event_uid: str
    title: str | None
    city: str | None
    first_timing_begin: str | None
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]


def resolve_allowed_cities() -> set[str]:
    settings = get_settings()
    configured = settings.openagenda_allowed_cities or settings.openagenda_city
    return {city.strip() for city in configured.split(",") if city.strip()}


@lru_cache(maxsize=1)
def get_rag_clients() -> tuple[MistralAIEmbeddings, ChatMistralAI]:
    settings = get_settings()
    if not settings.mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY is required to run the API.")
    embeddings = MistralAIEmbeddings(
        api_key=settings.mistral_api_key,
        model=settings.mistral_embedding_model,
    )
    chat_model = ChatMistralAI(
        api_key=settings.mistral_api_key,
        model_name=settings.mistral_chat_model,
        temperature=0.2,
    )
    return embeddings, chat_model


@app.get("/health")
def health() -> dict[str, object]:
    vector_store_dir = DEFAULT_VECTOR_STORE_DIR
    return {
        "status": "ok",
        "vector_store_exists": (vector_store_dir / "openagenda.faiss").exists()
        and (vector_store_dir / "openagenda_metadata.json").exists(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        embeddings, chat_model = get_rag_clients()
        result = answer_question(
            question=payload.question,
            embeddings=embeddings,
            chat_model=chat_model,
            vector_store_dir=Path(DEFAULT_VECTOR_STORE_DIR),
            top_k=payload.top_k,
            allowed_cities=resolve_allowed_cities(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceResponse(
                event_uid=source.chunk.event_uid,
                title=source.chunk.title,
                city=source.chunk.city,
                first_timing_begin=source.chunk.first_timing_begin,
                score=source.score,
            )
            for source in result.sources
        ],
    )
