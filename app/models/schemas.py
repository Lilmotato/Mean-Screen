# app/models/schemas.py

from enum import Enum
from typing import List
from pydantic import BaseModel, Field


# --- Core Enums ---

class ClassificationLabel(str, Enum):
    hate = "hate"
    toxic = "toxic"
    offensive = "offensive"
    neutral = "neutral"
    ambiguous = "ambiguous"


# --- Input Schema ---

class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="The user input text to be analyzed.")


# --- Agent Outputs ---

class ClassificationResult(BaseModel):
    label: ClassificationLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explanation of why the content was classified as such")


class PolicyDocument(BaseModel):
    id: str
    title: str
    content: str
    category: str
    relevance_score: float


class RetrievalResult(BaseModel):
    query_used: str
    policies: List[PolicyDocument]


# --- Final Output Schema ---

class AnalyzeResponse(BaseModel):
    label: ClassificationLabel
    confidence: float
    reasoning: str
    policies: List[str] = Field(..., description="Titles of relevant policy documents used for reasoning")
    explanation: str = Field(..., description="Natural language justification based on policy")
    recommendation: str = Field(..., description="Suggested moderation or escalation action")
