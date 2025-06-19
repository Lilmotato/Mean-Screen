# app/agents/retriever.py

import re
from typing import Dict, List

from app.agents.base import BaseAgent
from app.models.schemas import ClassificationResult, PolicyDocument, RetrievalResult
from app.services.embed_service import get_embedding_service
from app.services.qdrant_client import search_policies
from app.utils.exceptions import RetrievalError

# Constants
DEFAULT_TITLE = "Untitled"
DEFAULT_PROVIDER = "Unknown"
DEFAULT_TYPE = "general"


class HybridRetriever(BaseAgent):
    """
    Scores top policies based on:
    1. Semantic similarity
    2. Keyword match
    3. Policy type relevance
    """

    def __init__(self):
        super().__init__("hybrid_retriever")
        self.embedding_service = get_embedding_service()

    async def _execute(
        self, text: str, classification: ClassificationResult
    ) -> RetrievalResult:
        try:
            if not text or not isinstance(text, str):
                raise RetrievalError("Input text must be a non-empty string.")
            raw_results = search_policies(text, limit=8)

            if not raw_results:
                return RetrievalResult(policies=[], query_used=text, total_candidates=0)

            scored_results = self._score_and_explain(raw_results, text, classification)
            top_results = sorted(
                scored_results, key=lambda r: r["score"], reverse=True
            )[:3]

            policies = [
                PolicyDocument(
                    id=result["id"],
                    title=result["data"].get("title", DEFAULT_TITLE),
                    content=result["data"].get("content", ""),
                    category=result["data"].get("type", DEFAULT_TYPE),
                    relevance_score=result["score"],
                    source=result["data"].get("provider", DEFAULT_PROVIDER),
                    policy_type=result["data"].get("type", DEFAULT_TYPE),
                    explanation=result["explanation"],
                )
                for result in top_results
            ]

            return RetrievalResult(
                policies=policies, query_used=text, total_candidates=len(raw_results)
            )

        except Exception as e:
            raise RetrievalError(f"Hybrid retrieval failed: {e}")

    def _score_and_explain(
        self, results: List[Dict], text: str, classification: ClassificationResult
    ) -> List[Dict]:
        """
        Rerank raw results by combining:
        - Semantic similarity
        - Keyword relevance (reasoning + label keywords)
        - Policy type boost
        """
        reasoning_terms = self._extract_keywords(classification.reasoning)
        label_terms = {
            "hate": ["hate", "harassment", "discrimination"],
            "toxic": ["toxic", "abuse", "harmful"],
            "offensive": ["offensive", "slur", "inappropriate"],
        }.get(classification.label, [])

        for r in results:
            content = r["data"].get("content", "").lower()
            title = r["data"].get("title", "").lower()

            reasoning_score = (
                sum(1 for w in reasoning_terms if w in content or w in title) * 0.05
            )
            label_score = sum(1 for w in label_terms if w in content) * 0.1
            type_bonus = self._policy_type_boost(
                r["data"].get("type", ""), classification.label
            )

            final_score = r["score"] + reasoning_score + label_score + type_bonus
            r["score"] = min(final_score, 1.0)  # cap at 1.0
            r["explanation"] = self._generate_explanation(
                r["data"].get("title", DEFAULT_TITLE),
                r["data"].get("provider", DEFAULT_PROVIDER),
                reasoning_terms + label_terms,
                classification.label,
                final_score,
            )

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract significant keywords from reasoning text.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        stopwords = {
            "the",
            "and",
            "with",
            "for",
            "that",
            "this",
            "from",
            "into",
            "such",
        }
        return [w for w in words if len(w) > 3 and w not in stopwords]

    def _policy_type_boost(self, policy_type: str, label: str) -> float:
        """
        Return a score bonus based on policy type relevance for the classification label.
        """
        boost_map = {
            "hate": {"legal_framework": 0.15, "community_guidelines": 0.1},
            "toxic": {"community_guidelines": 0.15, "platform_policy": 0.1},
            "offensive": {"platform_policy": 0.1},
        }
        return boost_map.get(label, {}).get(policy_type, 0.0)

    def _generate_explanation(
        self,
        title: str,
        provider: str,
        matched_terms: List[str],
        label: str,
        score: float,
    ) -> str:
        """
        Generate a brief explanation string justifying the document's inclusion.
        """
        top_terms = (
            ", ".join(set(matched_terms[:3])) if matched_terms else "general relevance"
        )
        return (
            f"Matched '{title}' from {provider} due to {label} indicators "
            f"(terms: {top_terms}, score: {score:.2f})"
        )
