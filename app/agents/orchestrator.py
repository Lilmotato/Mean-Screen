import logging

from app.agents.classification_agent import ClassificationAgent
from app.agents.reasoner import PolicyReasoner
from app.agents.recommender import ActionRecommender
from app.agents.retriever import HybridRetriever
from app.agents.error_handler import ErrorHandlerAgent
from app.models.schemas import (ActionRecommendation, ActionType,
                                ConfidenceLevel, DetailedAnalyzeResponse,
                                HateSpeechClassification, PolicySummary,
                                SeverityLevel)
from app.services.llm_services import DIALService

logger = logging.getLogger(__name__)


class HateSpeechOrchestrator:
    """Coordinates classification, retrieval, reasoning, and action recommendation."""

    def __init__(self):
        llm_service = DIALService()
        self.detector = ClassificationAgent(llm_service)
        self.retriever = HybridRetriever()
        self.reasoner = PolicyReasoner(llm_service)
        self.recommender = ActionRecommender()
        self.error_handler = ErrorHandlerAgent()

    async def run(self, text: str) -> DetailedAnalyzeResponse:
        """Main pipeline for analyzing input text."""
        validation_result = self.error_handler.validate_input(text)
        if not validation_result["valid"]:
            raise ValueError(validation_result["message"])

        original_text = text.strip()
        logger.info(f"Processing text: '{original_text[:50]}...'")

        try:
            classification = await self.detector._execute(original_text)

            retrieval_result = await self.retriever._execute(
                original_text, classification
            )
            policies = retrieval_result.policies

            reasoning_output = await self.reasoner._execute(
                original_text, policies, classification
            )
            explanation = reasoning_output.get(
                "explanation", "No global explanation returned."
            )
            policy_explanations = reasoning_output.get("policy_summaries", {})

            recommendation = await self.recommender._execute(classification)

            for p in policies:
                p.explanation = policy_explanations.get(
                    p.id, "No specific summary provided."
                )

            return self._build_detailed_response(
                classification, policies, explanation, recommendation
            )

        except Exception as e:
            error_response = self.error_handler.handle_error(e, "orchestrator.run")
            logger.error(f"Orchestrator handled error: {error_response}")
            raise

    def _build_detailed_response(
        self, classification, policies, explanation, recommendation
    ) -> DetailedAnalyzeResponse:
        """Builds a DetailedAnalyzeResponse from agent outputs."""

        hate_speech_classification = HateSpeechClassification(
            classification=classification.label.capitalize(),
            confidence=self._get_confidence_level(classification.confidence),
            reason=classification.reasoning,
        )

        policy_summaries = [
            PolicySummary(
                source=p.source,
                summary=f"{p.title}: {p.content}\n{p.explanation}",
                relevance_score=round(p.relevance_score * 100, 2),
            )
            for p in policies
        ]

        return DetailedAnalyzeResponse(
            hate_speech=hate_speech_classification,
            policies=policy_summaries,
            reasoning=explanation,
            action=self._build_action_recommendation(classification),
        )

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _build_action_recommendation(self, classification) -> ActionRecommendation:
        label = classification.label
        mapping = {
            "hate": (
                ActionType.ESCALATE,
                SeverityLevel.HIGH,
                "Escalate due to serious hate speech.",
            ),
            "toxic": (
                ActionType.WARN,
                SeverityLevel.MEDIUM,
                "Warn user for toxic language.",
            ),
            "offensive": (
                ActionType.REVIEW,
                SeverityLevel.LOW,
                "Flag offensive content for review.",
            ),
            "ambiguous": (
                ActionType.REVIEW,
                SeverityLevel.LOW,
                "Ambiguous content—requires review.",
            ),
            "neutral": (
                ActionType.ALLOW,
                SeverityLevel.NONE,
                "Content is safe and can be allowed.",
            ),
        }
        action, severity, reasoning = mapping.get(
            label, (ActionType.ALLOW, SeverityLevel.NONE, "Content is safe.")
        )
        return ActionRecommendation(
            action=action, severity=severity, reasoning=reasoning
        )