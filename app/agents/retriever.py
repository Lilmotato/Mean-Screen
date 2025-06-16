from app.agents.base import BaseAgent
from app.services.qdrant_client import search_policies
from app.models.schemas import RetrievalResult, PolicyDocument, ClassificationResult
from app.utils.exceptions import RetrievalError

class HybridRetriever(BaseAgent):
    def __init__(self):
        super().__init__("hybrid_retriever")

    async def _execute(self, original_text: str, classification: ClassificationResult) -> RetrievalResult:
        try:
            query = self._build_query(original_text, classification)
            results = search_policies(query, limit=5)
            policies = [
                PolicyDocument(
                    id=r["id"],
                    title=r["data"].get("title", "Untitled Policy"),
                    content=r["data"].get("content", ""),
                    category=r["data"].get("category", "general"),
                    relevance_score=r["score"]
                )
                for r in results
            ]
            return RetrievalResult(policies=policies, query_used=query)
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}")

    def _build_query(self, text: str, classification: ClassificationResult) -> str:
        suffix = {
            "hate_speech": "hate speech violation policy enforcement",
            "borderline": "borderline content moderation guidelines"
        }.get(classification.label, "content policy guidelines")

        keywords = ["harassment", "discrimination", "threat", "violence",
                    "targeted", "protected", "group", "individual"]

        terms = [kw for kw in keywords if kw in classification.reasoning.lower()] if classification.reasoning else []
        return f"{text} {suffix} {' '.join(terms)}"
