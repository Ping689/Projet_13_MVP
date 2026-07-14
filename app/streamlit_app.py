from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def disable_broken_local_proxy() -> None:
    broken_proxy = "http://127.0.0.1:9"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if os.getenv(name) == broken_proxy:
            os.environ.pop(name, None)


disable_broken_local_proxy()

import streamlit as st
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from app.config import get_settings
from app.rag_chain import DEFAULT_VECTOR_STORE_DIR, answer_question_with_timings


def resolve_allowed_cities() -> set[str]:
    settings = get_settings()
    configured = settings.openagenda_allowed_cities or settings.openagenda_city
    return {city.strip() for city in configured.split(",") if city.strip()}


def get_rag_clients() -> tuple[MistralAIEmbeddings, ChatMistralAI]:
    settings = get_settings()
    if not settings.mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY est requis pour lancer le chatbot.")
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


def check_vector_store(vector_store_dir: Path) -> bool:
    return (vector_store_dir / "openagenda.faiss").exists() and (
        vector_store_dir / "openagenda_metadata.json"
    ).exists()


def render_sources(result_sources) -> None:
    if not result_sources:
        return

    with st.expander("Sources utilisées", expanded=False):
        for index, source in enumerate(result_sources, start=1):
            chunk = source.chunk
            title = chunk.title or "Titre inconnu"
            city = chunk.city or "Ville inconnue"
            date = chunk.first_timing_begin or "Date inconnue"
            st.markdown(f"**{index}. {title}**")
            st.caption(f"{city} | {date} | score={source.score:.3f}")


def render_timings(timings: dict[str, float]) -> None:
    displayed_total_seconds = (timings["embedding_seconds"] + timings["retrieval_seconds"] + timings["generation_seconds"])
    st.caption(
        " | ".join(
            [
                f"Embedding : {timings['embedding_seconds']:.2f} s",
                f"Recherche FAISS : {timings['retrieval_seconds']:.2f} s",
                f"Génération : {timings['generation_seconds']:.2f} s",
                f"Temps global : {displayed_total_seconds:.2f} s",
            ]
        )
    )


def main() -> None:
    st.set_page_config(page_title="Assistant sorties culturelles", page_icon="?", layout="centered")
    st.title("Assistant sorties culturelles")
    st.caption("Puls-Events vous aide à trouver des événements OpenAgenda à partir de l'index FAISS local.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Paramètres")
        top_k = st.slider("Sources à récupérer", min_value=1, max_value=10, value=5)
      
        vector_store_dir = DEFAULT_VECTOR_STORE_DIR

        allowed_cities = resolve_allowed_cities()
        st.caption(
            "Villes autorisées: "
            + (", ".join(sorted(allowed_cities)) if allowed_cities else "aucune restriction")
        )

        if check_vector_store(vector_store_dir):
            st.success("Index FAISS detecté.")
        else:
            st.error("Index FAISS introuvable.")

        if st.button("Réinitialiser la conversation"):
            del st.session_state["messages"]
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("timings"):
                render_timings(message["timings"])
            elif message.get("elapsed_seconds") is not None:
                st.caption(f"Temps global de traitement : {message['elapsed_seconds']:.2f} s")
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Posez une question sur les événements...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            embeddings, chat_model = get_rag_clients()
            with st.spinner("Recherche dans FAISS et génération de la réponse..."):
                result = answer_question_with_timings(
                    question=question,
                    embeddings=embeddings,
                    chat_model=chat_model,
                    vector_store_dir=vector_store_dir,
                    top_k=top_k,
                    allowed_cities=allowed_cities,
                )
        except Exception as exc:
            st.error(str(exc))
            return

        st.markdown(result.answer)
        timings = {
            "embedding_seconds": result.timings.embedding_seconds,
            "retrieval_seconds": result.timings.retrieval_seconds,
            "generation_seconds": result.timings.generation_seconds,
        }
        render_timings(timings)
        render_sources(result.sources)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
                "timings": timings,
            }
        )


if __name__ == "__main__":
    main()
