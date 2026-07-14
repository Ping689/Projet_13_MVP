from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from app.config import get_settings
from app.rag_chain import DEFAULT_VECTOR_STORE_DIR, RagAnswer, answer_question


DEFAULT_DATASET = ROOT_DIR / "data" / "eval" / "qa_dataset.json"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "eval" / "rag_eval_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évalue le chatbot RAG.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Fichier JSON du jeu d'évaluation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Fichier JSON de sortie des résultats.")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de chunks FAISS récupérés par question.")
    parser.add_argument("--vector-store-dir", type=Path, default=DEFAULT_VECTOR_STORE_DIR, help="Dossier de l'index FAISS.")
    return parser.parse_args()


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    normalized_answer = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in normalized_answer)
    return hits / len(expected_keywords)


def source_hit(answer: RagAnswer, expected_event_uids: list[str]) -> bool:
    retrieved_uids = {source.chunk.event_uid for source in answer.sources}
    return bool(retrieved_uids.intersection(expected_event_uids))


def evaluate_item(item: dict, answer: RagAnswer) -> dict:
    expected_event_uids = [str(uid) for uid in item.get("expected_event_uids", [])]
    return {
        "id": item["id"],
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "generated_answer": answer.answer,
        "source_hit": source_hit(answer, expected_event_uids),
        "keyword_coverage": keyword_coverage(answer.answer, item.get("expected_keywords", [])),
        "retrieved_sources": [
            {
                "event_uid": source.chunk.event_uid,
                "title": source.chunk.title,
                "city": source.chunk.city,
                "first_timing_begin": source.chunk.first_timing_begin,
                "score": source.score,
            }
            for source in answer.sources
        ],
    }


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY est requis pour lancer l'évaluation.")

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    embeddings = MistralAIEmbeddings(
        api_key=settings.mistral_api_key,
        model=settings.mistral_embedding_model,
    )
    chat_model = ChatMistralAI(
        api_key=settings.mistral_api_key,
        model_name=settings.mistral_chat_model,
        temperature=0.2,
    )

    results = []
    for item in dataset["items"]:
        answer = answer_question(
            question=item["question"],
            embeddings=embeddings,
            chat_model=chat_model,
            vector_store_dir=args.vector_store_dir,
            top_k=args.top_k,
        )
        results.append(evaluate_item(item, answer))

    source_hits = sum(1 for result in results if result["source_hit"])
    avg_keyword_coverage = sum(result["keyword_coverage"] for result in results) / len(results)
    payload = {
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": dataset["name"],
        "total_questions": len(results),
        "source_hit_rate": source_hits / len(results),
        "average_keyword_coverage": avg_keyword_coverage,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Évaluation enregistrée dans {args.output} | "
        f"source_hit_rate={payload['source_hit_rate']:.2f} | "
        f"average_keyword_coverage={payload['average_keyword_coverage']:.2f}"
    )


if __name__ == "__main__":
    main()
