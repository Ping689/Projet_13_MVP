import pytest
from app.vector_store import (
    get_db_connection,
    init_db_tables,
    add_chat_message,
    get_chat_history,
    add_user_feedback,
    similarity_search_by_vector,
    EventChunk
)

def test_database_connection_and_init():
    # Test de connexion et d'initialisation des tables
    conn = get_db_connection()
    assert conn is not None
    init_db_tables(conn)
    conn.close()

def test_conversational_memory():
    # Test d'insertion de messages et récupération de l'historique
    session_id = "test_session_123"
    
    # Nettoyage préalable des anciennes données de test
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chat_history WHERE session_id = %s;", (session_id,))
        conn.commit()
    conn.close()
    
    add_chat_message(session_id, "user", "Bonjour assistant !")
    add_chat_message(session_id, "assistant", "Bonjour ! Comment puis-je vous aider aujourd'hui ?")
    
    history = get_chat_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Bonjour assistant !"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Bonjour ! Comment puis-je vous aider aujourd'hui ?"

def test_user_feedback():
    # Test d'enregistrement du feedback utilisateur
    session_id = "test_session_123"
    message_id = "msg_001"
    
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM user_feedback WHERE session_id = %s;", (session_id,))
        conn.commit()
    conn.close()
    
    add_user_feedback(session_id, message_id, "thumbs_up", "Réponse très pertinente !")
    
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT rating, feedback_text FROM user_feedback WHERE session_id = %s;", (session_id,))
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "thumbs_up"
        assert row[1] == "Réponse très pertinente !"
    conn.close()

def test_geolocated_similarity_search():
    # Test du filtrage géographique avec la formule de Haversine
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM event_chunks WHERE chunk_id IN ('test:paris', 'test:bordeaux');")
        
        # Vecteur factice de dimension 1024
        dummy_vector = [0.0] * 1024
        
        # Insertion d'un événement à Paris
        cur.execute("""
            INSERT INTO event_chunks (chunk_id, event_uid, title, city, first_timing_begin, latitude, longitude, text, embedding)
            VALUES ('test:paris', 'evt_paris', 'Concert à Paris', 'Paris', NOW(), 48.8566, 2.3522, 'Un concert génial à Paris.', %s::vector);
        """, (dummy_vector,))
        
        # Insertion d'un événement à Bordeaux
        cur.execute("""
            INSERT INTO event_chunks (chunk_id, event_uid, title, city, first_timing_begin, latitude, longitude, text, embedding)
            VALUES ('test:bordeaux', 'evt_bordeaux', 'Exposition à Bordeaux', 'Bordeaux', NOW(), 44.8378, -0.5792, 'Une expo sympa à Bordeaux.', %s::vector);
        """, (dummy_vector,))
        
        conn.commit()
    conn.close()
    
    # Recherche à partir de Paris avec un rayon de 50 km (Bordeaux à 500+ km ne doit pas ressortir)
    results_50km = similarity_search_by_vector(
        query_vector=[0.0] * 1024,
        latitude=48.8566,
        longitude=2.3522,
        radius_km=50.0,
        top_k=2000
    )
    
    test_results_50 = [r for r in results_50km if r.chunk.chunk_id in ('test:paris', 'test:bordeaux')]
    assert len(test_results_50) == 1
    assert test_results_50[0].chunk.chunk_id == "test:paris"
    
    # Recherche à partir de Paris avec un rayon de 1000 km (les deux doivent ressortir)
    results_1000km = similarity_search_by_vector(
        query_vector=[0.0] * 1024,
        latitude=48.8566,
        longitude=2.3522,
        radius_km=1000.0,
        top_k=2000
    )
    
    test_results_1000 = [r for r in results_1000km if r.chunk.chunk_id in ('test:paris', 'test:bordeaux')]
    assert len(test_results_1000) == 2
