from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any
import unicodedata

import faiss
import numpy as np
import psycopg2
from app.config import get_settings


@dataclass(frozen=True)
class EventChunk:
    chunk_id: str
    event_uid: str
    title: str | None
    city: str | None
    first_timing_begin: str | None
    text: str
    latitude: float | None = None
    longitude: float | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk: EventChunk
    score: float


class EmbeddingsClient(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def get_db_connection():
    settings = get_settings()
    return psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password
    )


def init_db_tables(conn) -> None:
    with conn.cursor() as cur:
        # Activer l'extension pgvector
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Table des chunks vectoriels
        cur.execute("""
            CREATE TABLE IF NOT EXISTS event_chunks (
                chunk_id VARCHAR(100) PRIMARY KEY,
                event_uid VARCHAR(50),
                title TEXT,
                city TEXT,
                first_timing_begin TIMESTAMP,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                text TEXT,
                embedding vector(1024)
            );
        """)
        
        # Table pour la mémoire de session (Chatbot History)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Table pour le monitoring et feedbacks utilisateurs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id SERIAL PRIMARY KEY,
                message_id VARCHAR(100),
                session_id VARCHAR(100),
                rating VARCHAR(15), -- 'thumbs_up' / 'thumbs_down'
                feedback_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Tables requises par la persistance de l'interface Chainlit (SQLAlchemyDataLayer)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                identifier TEXT NOT NULL UNIQUE,
                metadata JSONB NOT NULL,
                "createdAt" TEXT
            );
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                id UUID PRIMARY KEY,
                "createdAt" TEXT,
                name TEXT,
                "userId" UUID,
                "userIdentifier" TEXT,
                tags TEXT[],
                metadata JSONB,
                FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
            );
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                id UUID PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                "threadId" UUID NOT NULL,
                "parentId" UUID,
                streaming BOOLEAN NOT NULL,
                "waitForAnswer" BOOLEAN,
                "isError" BOOLEAN,
                metadata JSONB,
                tags TEXT[],
                input TEXT,
                output TEXT,
                "createdAt" TEXT,
                command TEXT,
                start TEXT,
                "end" TEXT,
                generation JSONB,
                "showInput" TEXT,
                language TEXT,
                indent INT,
                "defaultOpen" BOOLEAN,
                modes JSONB,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            );
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS elements (
                id UUID PRIMARY KEY,
                "threadId" UUID,
                type TEXT,
                url TEXT,
                "chainlitKey" TEXT,
                name TEXT NOT NULL,
                display TEXT,
                "objectKey" TEXT,
                size TEXT,
                page INT,
                language TEXT,
                "forId" UUID,
                mime TEXT,
                props JSONB
            );
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedbacks (
                id UUID PRIMARY KEY,
                "forId" UUID,
                value INT,
                comment TEXT
            );
        """)
        conn.commit()


def load_processed_events(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    if not events:
        raise ValueError(f"No processed events found in {path}.")
    return events


def split_text(text: str, *, chunk_size: int = 900, chunk_overlap: int = 120) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap.")

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = end - chunk_overlap
    return chunks


def build_event_chunks(
    events: list[dict],
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[EventChunk]:
    chunks: list[EventChunk] = []
    for event in events:
        event_uid = str(event.get("uid"))
        lat = event.get("latitude")
        lon = event.get("longitude")
        
        # Conversion sécurisée en float
        try:
            latitude = float(lat) if lat is not None else None
            longitude = float(lon) if lon is not None else None
        except (ValueError, TypeError):
            latitude = None
            longitude = None

        for index, text in enumerate(
            split_text(event.get("text_for_rag", ""), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunks.append(
                EventChunk(
                    chunk_id=f"{event_uid}:{index}",
                    event_uid=event_uid,
                    title=event.get("title"),
                    city=event.get("city"),
                    first_timing_begin=event.get("first_timing_begin"),
                    text=text,
                    latitude=latitude,
                    longitude=longitude,
                )
            )
    if not chunks:
        raise ValueError("No text chunks were created from processed events.")
    return chunks


def embed_chunks(
    chunks: list[EventChunk],
    embeddings: EmbeddingsClient,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(embeddings.embed_documents([chunk.text for chunk in batch]))

    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError("Embedding output shape does not match the chunk count.")
    return matrix


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Vectors must be a non-empty 2D matrix.")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.replace("'", "-").replace(" ", "-").strip().lower()


def save_vector_store(
    *,
    index: Any = None,
    chunks: list[EventChunk],
    output_dir: Path = None,
    embedding_model: str = None,
) -> None:
    # Reconstitution des vecteurs depuis l'index FAISS si présent
    vectors = []
    if index is not None and hasattr(index, "reconstruct"):
        for i in range(index.ntotal):
            vectors.append(index.reconstruct(i))
    else:
        raise ValueError("save_vector_store requiert un index FlatIP ou des vecteurs valides.")

    conn = get_db_connection()
    try:
        init_db_tables(conn)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE event_chunks;")
            for chunk, vec in zip(chunks, vectors):
                vec_list = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
                cur.execute("""
                    INSERT INTO event_chunks (chunk_id, event_uid, title, city, first_timing_begin, latitude, longitude, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    chunk.chunk_id,
                    chunk.event_uid,
                    chunk.title,
                    chunk.city,
                    chunk.first_timing_begin,
                    chunk.latitude,
                    chunk.longitude,
                    chunk.text,
                    vec_list
                ))
            conn.commit()
    finally:
        conn.close()


def load_vector_store(vector_store_dir: Path = None) -> tuple[Any, list[EventChunk], dict]:
    # Simulation de la récupération de la liste des chunks depuis la DB
    conn = get_db_connection()
    chunks = []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id, event_uid, title, city, first_timing_begin, latitude, longitude, text FROM event_chunks;")
            rows = cur.fetchall()
            for r in rows:
                chunks.append(EventChunk(
                    chunk_id=r[0],
                    event_uid=r[1],
                    title=r[2],
                    city=r[3],
                    first_timing_begin=r[4].isoformat() if r[4] else None,
                    text=r[7],
                    latitude=r[5],
                    longitude=r[6]
                ))
    except Exception as e:
        print(f"Error loading vector store from DB: {e}")
    finally:
        conn.close()
    return None, chunks, {}


def similarity_search(
    *,
    query: str,
    index: Any = None,
    chunks: list[EventChunk] = None,
    embeddings: EmbeddingsClient,
    top_k: int = 5,
    allowed_cities: set[str] | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    radius_km: float | None = None,
) -> list[SearchResult]:
    query_vector = embeddings.embed_query(query)
    return similarity_search_by_vector(
        query_vector=query_vector,
        top_k=top_k,
        allowed_cities=allowed_cities,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km
    )


def similarity_search_by_vector(
    *,
    query_vector: np.ndarray | list[float],
    index: Any = None,
    chunks: list[EventChunk] = None,
    top_k: int = 5,
    allowed_cities: set[str] | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    radius_km: float | None = None,
) -> list[SearchResult]:
    if hasattr(query_vector, 'tolist'):
        query_vector = query_vector.tolist()
    
    # S'assurer que le vecteur est bien plat (1D)
    if isinstance(query_vector, list) and len(query_vector) > 0 and isinstance(query_vector[0], list):
        query_vector = query_vector[0]

    conn = get_db_connection()
    results = []
    try:
        with conn.cursor() as cur:
            filters = []
            params = [query_vector]

            if allowed_cities:
                normalized_cities = [normalize_text(c) for c in allowed_cities if c]
                if normalized_cities:
                    filters.append("LOWER(city) = ANY(%s)")
                    params.append(normalized_cities)

            if latitude is not None and longitude is not None and radius_km is not None:
                # Haversine distance formula in standard SQL
                filters.append("""
                    (6371 * acos(
                        least(1.0, greatest(-1.0, 
                            cos(radians(%s)) * cos(radians(latitude)) * cos(radians(longitude) - radians(%s)) 
                            + sin(radians(%s)) * sin(radians(latitude))
                        ))
                    )) <= %s
                """)
                params.extend([latitude, longitude, latitude, radius_km])

            where_clause = ""
            if filters:
                where_clause = "WHERE " + " AND ".join(filters)

            # Similarité cosinus avec pgvector (distance cosinus <=> triée en ordre croissant)
            query = f"""
                SELECT chunk_id, event_uid, title, city, first_timing_begin, text, latitude, longitude, (embedding <=> %s::vector) AS distance
                FROM event_chunks
                {where_clause}
                ORDER BY distance ASC
                LIMIT %s;
            """
            params.append(top_k)

            cur.execute(query, params)
            rows = cur.fetchall()
            for r in rows:
                chunk = EventChunk(
                    chunk_id=r[0],
                    event_uid=r[1],
                    title=r[2],
                    city=r[3],
                    first_timing_begin=r[4].isoformat() if r[4] else None,
                    text=r[5],
                    latitude=r[6],
                    longitude=r[7]
                )
                score = 1.0 - float(r[8])
                results.append(SearchResult(chunk=chunk, score=score))
    except Exception as e:
        print(f"Error in DB similarity search: {e}")
    finally:
        conn.close()
    return results


def get_chat_history(session_id: str, limit: int = 10) -> list[dict]:
    conn = get_db_connection()
    history = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT role, content FROM chat_history
                WHERE session_id = %s
                ORDER BY created_at ASC
                LIMIT %s;
            """, (session_id, limit))
            rows = cur.fetchall()
            for r in rows:
                history.append({"role": r[0], "content": r[1]})
    except Exception as e:
        print(f"Error fetching chat history from DB: {e}")
    finally:
        conn.close()
    return history


def add_chat_message(session_id: str, role: str, content: str) -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_history (session_id, role, content)
                VALUES (%s, %s, %s);
            """, (session_id, role, content))
            conn.commit()
    except Exception as e:
        print(f"Error saving chat message to DB: {e}")
    finally:
        conn.close()


def add_user_feedback(session_id: str, message_id: str, rating: str, feedback_text: str = None) -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_feedback (session_id, message_id, rating, feedback_text)
                VALUES (%s, %s, %s, %s);
            """, (session_id, message_id, rating, feedback_text))
            conn.commit()
    except Exception as e:
        print(f"Error saving user feedback to DB: {e}")
    finally:
        conn.close()

