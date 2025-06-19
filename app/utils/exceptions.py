"""
Custom exception classes for handling domain-specific errors
in classification, retrieval, policy loading, and agent orchestration.
"""

# ─── LLM/Service Exceptions ───────────────────────────────────────────


class LLMServiceError(Exception):
    """Raised when the LLM (e.g. DIAL/OpenAI) service fails or returns an invalid response."""

    pass


class PolicyLoadError(Exception):
    """Raised when policy documents fail to load or index properly."""

    pass


# ─── Agent-Orchestration Errors ───────────────────────────────────────


class AgentExecutionError(Exception):
    """Raised when a general agent execution step fails."""

    pass


class ClassificationError(Exception):
    """Raised when text classification fails or response is malformed."""

    pass


class RetrievalError(Exception):
    """Raised when the policy retrieval process fails."""

    pass


class ReasoningError(Exception):
    """Raised when policy reasoning/explanation fails."""

    pass


class RecommendationError(Exception):
    """Raised when the action recommender fails to produce output."""

    pass


class TranscriptionError(Exception):
    """Raised when audio-to-text transcription fails."""

    pass
