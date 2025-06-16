# app/agents/recommender.py
from app.agents.base import BaseAgent
from app.models.schemas import ClassificationResult
from app.utils.exceptions import AgentExecutionError

class ActionRecommender(BaseAgent):
    def __init__(self):
        super().__init__("action_recommender")

    async def _execute(self, classification: ClassificationResult) -> str:
        try:
            label = classification.label
            severity = classification.confidence  # optionally, use severity_score if available

            if label == "hate":
                return "Escalate to human moderator and restrict account"
            elif label == "toxic":
                return "Issue warning and monitor behavior"
            elif label == "offensive":
                return "Flag for review"
            elif label == "ambiguous":
                return "Send for manual review"
            else:  # neutral
                return "No action needed"
        except Exception as e:
            raise AgentExecutionError(f"Failed to recommend action: {e}")
