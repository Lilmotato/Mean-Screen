import pytest
from app.agents.error_handler import ErrorHandlerAgent
from app.utils.exceptions import (
    AgentExecutionError,
    ClassificationError,
    RetrievalError,
)


def test_validate_input_empty():
    handler = ErrorHandlerAgent()
    result = handler.validate_input("")
    assert result["valid"] is False
    assert "Empty input" in result["error"]


def test_validate_input_too_short():
    handler = ErrorHandlerAgent()
    result = handler.validate_input("ok")
    assert result["valid"] is False
    assert "Input too short" in result["error"]


def test_validate_input_too_long():
    handler = ErrorHandlerAgent()
    result = handler.validate_input("x" * 5001)
    assert result["valid"] is False
    assert "Input too long" in result["error"]


def test_validate_input_success():
    handler = ErrorHandlerAgent()
    result = handler.validate_input("This is a valid test input.")
    assert result["valid"] is True


@pytest.mark.parametrize(
    "error, expected_type",
    [
        (ClassificationError("LLM classification failed"), "Classification Error"),
        (RetrievalError("Qdrant retrieval failed"), "Policy Retrieval Error"),
        (AgentExecutionError("Something went wrong"), "Agent Error"),
        (Exception("llm timed out"), "AI Service Error"),
        (Exception("unhandled crash"), "System Error"),
    ],
)
def test_handle_error_classification(error, expected_type):
    handler = ErrorHandlerAgent()
    result = handler.handle_error(error, context="test_context")

    assert result["error"] is True
    assert result["type"] == expected_type
    assert isinstance(result["message"], str)
    assert isinstance(result["suggestion"], str)
