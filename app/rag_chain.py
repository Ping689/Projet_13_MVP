from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.vector_store import (
    EmbeddingsClient,
    SearchResult,
    similarity_search,
    similarity_search_by_vector,
    get_chat_history,
    add_chat_message,
)
from app.agent_search import search_web_for_events

DEFAULT_VECTOR_STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "vector_store"


class ChatModel(Protocol):
    def invoke(self, input: Any) -> Any:
        ...


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    sources: list[SearchResult]
    source_type: str = "database"  # 'database', 'web' ou 'empty'


@dataclass(frozen=True)
class RagTimings:
    embedding_seconds: float
    retrieval_seconds: float
    generation_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.embedding_seconds + self.retrieval_seconds + self.generation_seconds


@dataclass(frozen=True)
class TimedRagAnswer:
    question: str
    answer: str
    sources: list[SearchResult]
    timings: RagTimings
    source_type: str = "database"


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """Tu es le chatbot RAG de Puls-Events, spécialisé dans les événements culturels.
Si la question n'a aucun rapport avec la culture, les arts, les loisirs, les sorties ou les événements, réponds :
"Je suis spécialisé dans les événements culturels. Je ne peux pas répondre à cette question."

Règles:
- Réponds en français, de façon utile et concise. 
- Utilise uniquement les informations présentes dans le contexte.
- Si le contexte ne contient aucun événement pertinent pour la ville ou le sujet demandé, réponds clairement que tu n'as pas trouvé d'événements correspondants dans les sources disponibles.
- Lorsque le contexte provient de la recherche web, si des artistes ou des événements pertinents pour 2026 sont mentionnés sans dates précises ou détails complets (ex: Sean Paul ou Sigur Rós à Lyon), présente-les quand même dans le format requis en indiquant 'Non précisé' ou '2026' pour les champs manquants, au lieu de dire que tu n'as rien trouvé.
- N'invente jamais d'information.
- Respecte strictement les dates et périodes demandées.
- Ne recommande pas d'évènement hors période.
- Utilise les événements présents dans le contexte dès qu'ils correspondent approximativement à la demande.
- Si l'utilisateur demande une ville située hors du périmètre des sources fournies, indique-le clairement.

Pour chaque événement recommandé, utilise le format Markdown suivant :

1. Titre de l'événement

- **Ville :** ...
- **Date :** ...
- **Lieu :** ...
- **Tarif :** ...
- **Public visé :** ...
- **Description :**
...
Courte description de 2 à 3 phrases maximum.
---

Important:
- Laisse une ligne vide entre chaque champ.
- Laisse deux lignes vides entre deux événements.
- N'écris jamais plusieurs informations sur la même ligne.
- N'affiche jamais toutes les informations sur une seule ligne.
- Affiche au maximum {top_k} évènements pertinents.
- La réponse doit être facile à lire dans une interface conversationnelle Chat (Chainlit).
- N'utilise pas de tableau.
- N'utilise pas d'emoji.

Historique récent de la conversation :
{chat_history}
"""
        ),
        (
            "human",
            "Question utilisateur:\n{question}\n\nNombre d'événements à afficher si possible : {top_k}\n\n"
            "Contexte extrait :\n{context}",
        ),
    ]
)


def format_context(results: list[SearchResult]) -> str:
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        header = (
            f"Source {index} | score={result.score:.3f} | "
            f"titre={chunk.title or 'inconnu'} | ville={chunk.city or 'inconnue'} | "
            f"date={chunk.first_timing_begin or 'inconnue'} | uid={chunk.event_uid}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)

def rewrite_query_for_search(question: str, allowed_cities: set[str] | None, chat_model: ChatModel) -> str:
    current_year = "2026"
    prompt = (
        "Tu es un assistant de recherche. Rédige une courte requête de recherche web en français "
        f"pour trouver des événements en {current_year} correspondant à la question de l'utilisateur.\n"
        "Règles :\n"
        "- Sois court et direct (ex: 'expositions à Tours 2026' ou 'activités pour enfants à Tours 2026').\n"
        "- Inclus les prépositions naturelles comme 'à' ou 'pour' si nécessaire pour le français.\n"
        "- N'écris pas de phrase complète (pas de verbes conjugués), pas de ponctuation.\n"
        "- Évite les mots vagues comme 'actualité', 'culturel', 'proposées', 'calendrier', 'guide'.\n"
        f"Question : {question}\n"
        f"Villes cibles : {', '.join(allowed_cities) if allowed_cities else 'aucune spécifiée'}\n"
        "Requête de recherche :"
    )
    try:
        response = chat_model.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        cleaned = content.strip().replace('"', '').replace("'", "")
        return cleaned
    except Exception as e:
        print(f"[Query Rewriter] Error: {e}")
        return question


def answer_question(
    *,
    question: str,
    embeddings: EmbeddingsClient,
    chat_model: ChatModel,
    vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR,
    top_k: int = 5,
    allowed_cities: set[str] | None = None,
    session_id: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    radius_km: float | None = None,
) -> RagAnswer:
    # 1. Récupération de l'historique conversationnel
    history = []
    if session_id:
        history = get_chat_history(session_id, limit=6)
    
    formatted_history = ""
    if history:
        formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

    # 2. Vérification si l'utilisateur demande explicitement le web
    force_web = any(kw in question.lower() for kw in ["web", "internet", "en ligne", "recherche direct", "google", "ddg"])
    
    sources = []
    source_type = "database"
    context = ""

    if force_web:
        print("[RAG] Force Web search triggered.")
        web_results = search_web_for_events(question)
        context = f"Résultats de recherche en direct sur le web :\n{web_results}"
        source_type = "web"
    else:
        # Recherche vectorielle dans PostgreSQL local
        sources = similarity_search(
            query=question,
            embeddings=embeddings,
            top_k=top_k,
            allowed_cities=allowed_cities,
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
        )

        # Déclenchement du fallback web si aucun résultat local n'est pertinent
        use_web_fallback = False
        if not sources:
            use_web_fallback = True
        elif all(s.score < 0.25 for s in sources):
            use_web_fallback = True

        if use_web_fallback:
            print("[RAG] No relevant database sources found. Falling back to Web Search.")
            web_query = rewrite_query_for_search(question, allowed_cities, chat_model)
            print(f"[RAG] Web query rewritten: '{web_query}'")
            web_results = search_web_for_events(web_query)
            context = f"Résultats de recherche web en direct (repli car pas de source locale pertinente) :\n{web_results}"
            source_type = "web"
        else:
            context = format_context(sources)
            source_type = "database"

    chain = PROMPT | chat_model | StrOutputParser()
    answer = chain.invoke({
        "question": question,
        "top_k": top_k,
        "context": context,
        "chat_history": formatted_history
    })

    # 3. Enregistrement du message dans l'historique
    if session_id:
        add_chat_message(session_id, "user", question)
        add_chat_message(session_id, "assistant", answer)

    return RagAnswer(question=question, answer=answer, sources=sources, source_type=source_type)


def answer_question_with_timings(
    *,
    question: str,
    embeddings: EmbeddingsClient,
    chat_model: ChatModel,
    vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR,
    top_k: int = 5,
    allowed_cities: set[str] | None = None,
    session_id: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    radius_km: float | None = None,
) -> TimedRagAnswer:
    # 1. Récupération de l'historique conversationnel
    history = []
    if session_id:
        history = get_chat_history(session_id, limit=6)
    
    formatted_history = ""
    if history:
        formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

    force_web = any(kw in question.lower() for kw in ["web", "internet", "en ligne", "recherche direct", "google", "ddg"])

    # Étape 1 : Latence de l'embedding
    start_time = time.perf_counter()
    query_vector = np.asarray([embeddings.embed_query(question)], dtype="float32")
    embedding_seconds = time.perf_counter() - start_time

    # Étape 2 : Latence de la récupération
    start_time = time.perf_counter()
    sources = []
    source_type = "database"
    context = ""

    if force_web:
        web_query = rewrite_query_for_search(question, allowed_cities, chat_model)
        print(f"[RAG] Web query rewritten (force_web): '{web_query}'", flush=True)
        web_results = search_web_for_events(web_query)
        context = f"Résultats de recherche en direct sur le web :\n{web_results}"
        source_type = "web"
        retrieval_seconds = time.perf_counter() - start_time
    else:
        sources = similarity_search_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            allowed_cities=allowed_cities,
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
        )
        retrieval_seconds = time.perf_counter() - start_time

        use_web_fallback = False
        if not sources:
            use_web_fallback = True
        elif all(s.score < 0.25 for s in sources):
            use_web_fallback = True

        if use_web_fallback:
            fallback_start = time.perf_counter()
            web_query = rewrite_query_for_search(question, allowed_cities, chat_model)
            print(f"[RAG] Web query rewritten (fallback): '{web_query}'", flush=True)
            web_results = search_web_for_events(web_query)
            context = f"Résultats de recherche web en direct (repli car pas de source locale pertinente) :\n{web_results}"
            source_type = "web"
            retrieval_seconds += (time.perf_counter() - fallback_start)
        else:
            context = format_context(sources)
            source_type = "database"

    # Étape 3 : Latence de la génération
    start_time = time.perf_counter()
    chain = PROMPT | chat_model | StrOutputParser()
    answer = chain.invoke({
        "question": question,
        "top_k": top_k,
        "context": context,
        "chat_history": formatted_history
    })
    generation_seconds = time.perf_counter() - start_time

    # Enregistrement du message dans l'historique
    if session_id:
        add_chat_message(session_id, "user", question)
        add_chat_message(session_id, "assistant", answer)

    timings = RagTimings(
        embedding_seconds=embedding_seconds,
        retrieval_seconds=retrieval_seconds,
        generation_seconds=generation_seconds
    )

    return TimedRagAnswer(
        question=question,
        answer=answer,
        sources=sources,
        source_type=source_type,
        timings=timings
    )
