from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from app.config import get_settings
from app.rag_chain import DEFAULT_VECTOR_STORE_DIR, answer_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose des questions au chatbot RAG OpenAgenda.")
    parser.add_argument("--question", default=None, help="Question à poser. Si absente, lance le mode interactif.")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de chunks FAISS à récupérer.")
    parser.add_argument("--vector-store-dir", type=Path, default=DEFAULT_VECTOR_STORE_DIR, help="Dossier de l'index FAISS.")
    return parser.parse_args()


def build_clients() -> tuple[MistralAIEmbeddings, ChatMistralAI]:
    settings = get_settings()
    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY est requis pour lancer le chatbot.")
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


def resolve_allowed_cities() -> set[str]:
    settings = get_settings()
    configured = settings.openagenda_allowed_cities or settings.openagenda_city
    return {city.strip() for city in configured.split(",") if city.strip()}


def print_answer(question: str, *, top_k: int, vector_store_dir: Path) -> None:
    embeddings, chat_model = build_clients()
    result = answer_question(
        question=question,
        embeddings=embeddings,
        chat_model=chat_model,
        vector_store_dir=vector_store_dir,
        top_k=top_k,
        allowed_cities=resolve_allowed_cities(),
    )
    print("\nRéponse:")
    print(result.answer)
    print("\nSources:")
    for source in result.sources:
        chunk = source.chunk
        print(f"- {chunk.title} | {chunk.city} | {chunk.first_timing_begin} | score={source.score:.3f}")


def main() -> None:
    args = parse_args()
    if args.question:
        print_answer(args.question, top_k=args.top_k, vector_store_dir=args.vector_store_dir)
        return

    print("Chatbot RAG OpenAgenda. Tapez 'exit', 'quit' ou 'q' pour quitter.")
    while True:
        question = input("\nQuestion > ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        print_answer(question, top_k=args.top_k, vector_store_dir=args.vector_store_dir)


if __name__ == "__main__":
    main()
