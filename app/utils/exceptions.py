class ClassificationError(Exception):
    """Raised when classification fails or response is invalid"""
    pass

class LLMServiceError(Exception):
    """Raised for LLM service errors"""
    pass

class AgentExecutionError(Exception):
    """Raised for general agent errors"""
    pass


class RetrievalError(Exception):
    """Raised for general agent errors"""
    pass