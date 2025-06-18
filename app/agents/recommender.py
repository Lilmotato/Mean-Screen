# app/agents/recommender.py
from app.agents.base import BaseAgent
from app.models.schemas import ClassificationResult, ActionType, SeverityLevel
from app.utils.exceptions import AgentExecutionError

class ActionRecommender(BaseAgent):
    def __init__(self):
        super().__init__("action_recommender")

    async def _execute(self, classification: ClassificationResult) -> dict:
        """Return structured action recommendation"""
        try:
            label = classification.label
            confidence = classification.confidence

            # Determine action and severity based on classification and confidence
            if label == "hate":
                if confidence >= 0.8:
                    action = ActionType.ESCALATE
                    severity = SeverityLevel.HIGH
                    reasoning = "High confidence hate speech detection requires immediate escalation and account restrictions."
                else:
                    action = ActionType.REVIEW
                    severity = SeverityLevel.MEDIUM
                    reasoning = "Potential hate speech requires human review due to lower confidence score."
                    
            elif label == "toxic":
                if confidence >= 0.7:
                    action = ActionType.WARN
                    severity = SeverityLevel.MEDIUM
                    reasoning = "Toxic content warrants user warning and behavior monitoring."
                else:
                    action = ActionType.REVIEW
                    severity = SeverityLevel.LOW
                    reasoning = "Potentially toxic content needs review due to confidence level."
                    
            elif label == "offensive":
                action = ActionType.REVIEW
                severity = SeverityLevel.LOW
                reasoning = "Offensive content requires human review for context and appropriate action."
                
            elif label == "ambiguous":
                action = ActionType.REVIEW
                severity = SeverityLevel.LOW
                reasoning = "Ambiguous classification requires manual review for proper determination."
                
            else:  # neutral
                action = ActionType.ALLOW
                severity = SeverityLevel.NONE
                reasoning = "Content is neutral and complies with community guidelines."

            return {
                "action": action,
                "severity": severity,
                "reasoning": reasoning
            }
            
        except Exception as e:
            raise AgentExecutionError(f"Failed to recommend action: {e}")

    def get_simple_recommendation(self, classification: ClassificationResult) -> str:
        """Legacy method for backward compatibility - returns simple string recommendation"""
        try:
            label = classification.label
            confidence = classification.confidence

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