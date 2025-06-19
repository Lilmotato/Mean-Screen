# app/models/schemas.py

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class ClassificationLabel(str, Enum):
    hate = "hate"
    toxic = "toxic"
    offensive = "offensive"
    neutral = "neutral"
    ambiguous = "ambiguous"


class ConfidenceLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ActionType(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    REMOVE = "REMOVE"
    ESCALATE = "ESCALATE"
    REVIEW = "REVIEW"


class SeverityLevel(str, Enum):
    NONE = "None"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="The user input text to be analyzed.")


class ClassificationResult(BaseModel):
    label: ClassificationLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(
        ..., description="Explanation of why the content was classified as such"
    )


class PolicyDocument(BaseModel):
    id: str
    title: str
    content: str
    category: str
    relevance_score: float
    source: str = Field(
        ..., description="Origin of the policy document (e.g., platform, legal body)"
    )
    policy_type: str = Field(
        ...,
        description="Type of policy document such as community_guidelines or legal_framework",
    )
    explanation: str = Field(
        ...,
        description="Why this policy was considered relevant for the classification",
    )


class RetrievalResult(BaseModel):
    query_used: str
    policies: List[PolicyDocument]
    total_candidates: int = Field(
        ...,
        description="Number of total candidates initially retrieved before reranking",
    )


class HateSpeechClassification(BaseModel):
    classification: str = Field(
        ...,
        description="Classification result (Hate, Toxic, Offensive, Neutral, Ambiguous)",
    )
    confidence: ConfidenceLevel = Field(
        ..., description="Confidence level of classification"
    )
    reason: str = Field(..., description="Brief explanation of the classification")


class PolicySummary(BaseModel):
    source: str = Field(..., description="Policy source (e.g., Meta, Reddit, Google)")
    summary: str = Field(..., description="Brief summary of how the policy applies")
    relevance_score: float = Field(..., description="Relevance score as percentage")


class ActionRecommendation(BaseModel):
    action: ActionType = Field(..., description="Recommended action to take")
    severity: SeverityLevel = Field(..., description="Severity level of the content")
    reasoning: str = Field(..., description="Justification for the recommended action")


class DetailedAnalyzeResponse(BaseModel):
    hate_speech: HateSpeechClassification
    policies: List[PolicySummary] = Field(
        ..., description="Relevant policies with summaries"
    )
    reasoning: str = Field(
        ..., description="Overall reasoning based on policy analysis"
    )
    action: ActionRecommendation = Field(
        ..., description="Recommended moderation action"
    )
