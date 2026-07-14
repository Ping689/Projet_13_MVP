import sys
from pathlib import Path
ROOT_DIR = Path.cwd()
sys.path.insert(0, str(ROOT_DIR))
print(f"[DEBUG] ROOT_DIR determined as: {ROOT_DIR}")
print(f"[DEBUG] sys.path top: {sys.path[:3]}")

import chainlit as cl
from chainlit.input_widget import Slider, TextInput
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from app.config import get_settings

settings = get_settings()
DB_URL = f"postgresql+asyncpg://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=DB_URL)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Permet de se connecter avec n'importe quel nom d'utilisateur (ex: Pascal)
    return cl.User(identifier=username)


@cl.on_chat_start
async def start():
    import sys
    from pathlib import Path
    ROOT_DIR = Path.cwd()
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    print(f"[DEBUG START] sys.path updated: {sys.path[:4]}")
    # Configuration des widgets de réglage géographiques et de RAG dans la barre latérale
    await cl.ChatSettings([
        Slider(id="radius_km", label="Rayon de recherche géographique (km)", min=1, max=100, step=1, initial=20),
        Slider(id="top_k", label="Nombre d'événements (top_k)", min=1, max=10, step=1, initial=5),
        TextInput(id="city", label="Filtrer par ville (ex: Paris)", initial="Paris"),
        TextInput(id="gps_coords", label="Position GPS (latitude, longitude)", initial="48.874156, 2.316734")
    ]).send()
    
    # Initialisation des variables de session
    cl.user_session.set("radius_km", 20)
    cl.user_session.set("top_k", 5)
    cl.user_session.set("city", "Paris")
    cl.user_session.set("latitude", 48.874156)
    cl.user_session.set("longitude", 2.316734)
    cl.user_session.set("session_id", cl.user_session.get("id"))
    
    # Initialisation de la base de données PostgreSQL locale si nécessaire
    try:
        from app.vector_store import get_db_connection, init_db_tables
        conn = get_db_connection()
        init_db_tables(conn)
        conn.close()
    except Exception as e:
        print(f"[DB INIT] Erreur lors de l'initialisation de la DB locale : {e}")

    await cl.Message(
        content="✨ **Bienvenue sur le MVP de Puls-Events !**\n\n"
                "Je suis votre assistant IA spécialisé dans la recommandation d'événements culturels.\n\n"
                "📍 **Géolocalisation :** Par défaut, je recherche dans un rayon de **20 km** autour de **Paris**.\n"
                "⚙️ Vous pouvez modifier ces critères (GPS, ville, rayon de recherche, top_k) à tout moment dans les **paramètres du chat** situés dans la barre de gauche.\n\n"
                "Posez-moi votre question !"
    ).send()


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("radius_km", settings["radius_km"])
    cl.user_session.set("top_k", settings["top_k"])
    cl.user_session.set("city", settings["city"])
    
    # Extraction et nettoyage des coordonnées GPS
    gps = settings["gps_coords"]
    try:
        lat_str, lon_str = gps.split(",")
        cl.user_session.set("latitude", float(lat_str.strip()))
        cl.user_session.set("longitude", float(lon_str.strip()))
    except Exception:
        pass


@cl.on_message
async def main(message: cl.Message):
    # Récupération des paramètres utilisateur depuis la session
    radius_km = cl.user_session.get("radius_km")
    top_k = cl.user_session.get("top_k")
    city = cl.user_session.get("city")
    latitude = cl.user_session.get("latitude")
    longitude = cl.user_session.get("longitude")
    session_id = cl.user_session.get("session_id")
    
    # Transformation de la ville en ensemble pour le filtre exact
    allowed_cities = None
    if city and city.strip():
        allowed_cities = {c.strip() for c in city.split(",") if c.strip()}
        
    # Import dynamique des clients et de la chaîne RAG
    import sys
    from pathlib import Path
    ROOT_DIR = Path.cwd()
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    print(f"[DEBUG MAIN] sys.path updated: {sys.path[:4]}")
    from scripts.chatbot_demo import build_clients
    from app.rag_chain import answer_question_with_timings
    
    # Création du message d'attente
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        embeddings, chat_model = build_clients()
        
        # Exécution du pipeline RAG avec calcul de temps et historique SQL
        result = answer_question_with_timings(
            question=message.content,
            embeddings=embeddings,
            chat_model=chat_model,
            top_k=top_k,
            allowed_cities=allowed_cities,
            session_id=session_id,
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km
        )
        
        # Formatage de la réponse
        answer_text = result.answer
        
        # Ajout des informations de monitoring de performance
        timings = result.timings
        performance_info = (
            f"\n\n---\n"
            f"*⏱️ Performance MVP : Embedding={timings.embedding_seconds:.2f}s | "
            f"Recherche/Retrieval={timings.retrieval_seconds:.2f}s | "
            f"Génération LLM={timings.generation_seconds:.2f}s | "
            f"Total={timings.total_seconds:.2f}s | "
            f"Source={result.source_type}*"
        )
        msg.content = answer_text + performance_info
        await msg.update()
        
        # Si des sources documentaires locales ont été trouvées, les lister en éléments détachés
        if result.sources:
            elements = []
            for idx, source in enumerate(result.sources, start=1):
                chunk = source.chunk
                elements.append(cl.Text(
                    name=f"Source {idx} : {chunk.title}",
                    content=(
                        f"**Titre :** {chunk.title}\n"
                        f"**Ville :** {chunk.city}\n"
                        f"**Date :** {chunk.first_timing_begin}\n"
                        f"**Score de pertinence :** {source.score:.3f}\n\n"
                        f"**Extrait du document :**\n{chunk.text}"
                    ),
                    display="side"
                ))
            msg.elements = elements
            await msg.update()
            
    except Exception as e:
        msg.content = f"❌ Une erreur s'est produite lors de la génération de la réponse : {e}"
        await msg.update()
