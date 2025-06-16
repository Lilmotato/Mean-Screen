# app/agents/classification_agent.py
from app.agents.base import BaseAgent
from app.services.llm_services import DIALService
from app.models.schemas import ClassificationResult, ClassificationLabel
from app.utils.exceptions import ClassificationError

class ClassificationAgent(BaseAgent):
    def __init__(self, llm_service: DIALService):
        super().__init__("classification_agent")
        self.llm_service = llm_service

    async def _execute(self, text: str) -> ClassificationResult:
        try:
            result = await self.llm_service.classify_text(text)
            return ClassificationResult(
                label=ClassificationLabel(result["label"]),
                confidence=float(result["confidence"]),
                reasoning=result["reasoning"]
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ClassificationError(f"Invalid LLM response: {e}")
