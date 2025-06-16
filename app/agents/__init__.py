# app/agents/__init__.py
from app.agents.classification_agent import ClassificationAgent
from app.services.llm_services import DIALService

llm_service = DIALService()

agents = {
    "classifier": ClassificationAgent(llm_service),
    # "rag": RAGAgent(...),
    # "reasoner": ReasoningAgent(...),
    # "recommender": ActionRecommender(...),
    # "error_handler": ErrorHandlerAgent(...)
}
