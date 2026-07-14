from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, OpenAIServerModel, VisitWebpageTool
from app.config import get_settings

def get_web_search_agent():
    settings = get_settings()
    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY est requis dans le fichier .env pour utiliser l'agent de recherche.")

    # Utilisation du endpoint compatible OpenAI de Mistral AI
    model = OpenAIServerModel(
        model_id=settings.mistral_chat_model,
        api_base="https://api.mistral.ai/v1",
        api_key=settings.mistral_api_key
    )
    
    # Initialisation de l'outil de recherche
    search_tool = DuckDuckGoSearchTool()
    
    # Création de l'agent avec limite d'étapes pour éviter les boucles infinies
    agent = ToolCallingAgent(
        tools=[search_tool],
        model=model,
        max_steps=3
    )
    
    return agent

def search_web_for_events(query: str) -> str:
    try:
        from smolagents import DuckDuckGoSearchTool
        tool = DuckDuckGoSearchTool()
        results = tool(query)
        return str(results)
    except Exception as e:
        return f"Erreur lors de la recherche web en direct : {e}"
