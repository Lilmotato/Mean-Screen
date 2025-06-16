# app/agents/orchestrator.py
from app.agents.classification_agent import ClassificationAgent
from app.agents.reasoner import PolicyReasoner
from app.agents.recommender import ActionRecommender
from app.agents.retriever import HybridRetriever
from app.models.schemas import AnalyzeResponse
from app.services.llm_services import DIALService

class HateSpeechOrchestrator:
    def __init__(self):
        llm_service = DIALService()
        self.detector = ClassificationAgent(llm_service)
        self.retriever = HybridRetriever()
        self.reasoner = PolicyReasoner(llm_service)
        self.recommender = ActionRecommender()

    async def run(self, text: str) -> AnalyzeResponse:
        classification = await self.detector._execute(text)
        retrieval_result = await self.retriever._execute(text, classification)
        policies = retrieval_result.policies if hasattr(retrieval_result, 'policies') else retrieval_result
        explanation = await self.reasoner._execute(text, policies, classification)
        recommendation = await self.recommender._execute(classification)

        return AnalyzeResponse(
            label=classification.label,
            confidence=classification.confidence,
            reasoning=classification.reasoning,
            policies=[p.title for p in policies],
            explanation=explanation,
            recommendation=recommendation
        )
