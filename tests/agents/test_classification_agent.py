import pytest
from app.agents.classification_agent import ClassificationAgent
from app.models.schemas import ClassificationLabel, ClassificationResult
from app.utils.exceptions import ClassificationError


@pytest.mark.asyncio
async def test_execute_valid_response(mocker):
    # Arrange
    mock_llm = mocker.Mock()
    mock_llm.classify_text = mocker.AsyncMock(
        return_value={
            "label": "hate",
            "confidence": "0.92",
            "reasoning": "Contains slurs or threats.",
        }
    )

    agent = ClassificationAgent(llm_service=mock_llm)

    # Act
    result = await agent._execute("You are terrible and should leave.")

    # Assert
    assert isinstance(result, ClassificationResult)
    assert result.label == ClassificationLabel.hate
    assert result.confidence == 0.92
    assert "slurs" in result.reasoning.lower()


@pytest.mark.asyncio
async def test_execute_missing_key_raises_error(mocker):
    mock_llm = mocker.Mock()
    mock_llm.classify_text = mocker.AsyncMock(
        return_value={
            "confidence": "0.95",  # 'label' missing
            "reasoning": "Ambiguous language",
        }
    )

    agent = ClassificationAgent(llm_service=mock_llm)

    with pytest.raises(ClassificationError, match="Invalid LLM response"):
        await agent._execute("test input")


@pytest.mark.asyncio
async def test_execute_invalid_confidence_type_raises_error(mocker):
    mock_llm = mocker.Mock()
    mock_llm.classify_text = mocker.AsyncMock(
        return_value={
            "label": "neutral",
            "confidence": None,  # Invalid type
            "reasoning": "Neutral tone",
        }
    )

    agent = ClassificationAgent(llm_service=mock_llm)

    with pytest.raises(ClassificationError, match="Invalid LLM response"):
        await agent._execute("test input")


@pytest.mark.asyncio
async def test_execute_invalid_label_raises_error(mocker):
    mock_llm = mocker.Mock()
    mock_llm.classify_text = mocker.AsyncMock(
        return_value={
            "label": "unknown",  # Not in ClassificationLabel enum
            "confidence": "0.80",
            "reasoning": "Does not match known categories",
        }
    )

    agent = ClassificationAgent(llm_service=mock_llm)

    with pytest.raises(ClassificationError, match="Invalid LLM response"):
        await agent._execute("input text")
