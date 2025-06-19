import pytest
from app.agents.recommender import ActionRecommender
from app.models.schemas import ClassificationResult, ClassificationLabel
from app.models.schemas import ActionType, SeverityLevel
from app.utils.exceptions import AgentExecutionError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label,confidence,expected_action,expected_severity",
    [
        ("hate", 0.9, ActionType.ESCALATE, SeverityLevel.HIGH),
        ("hate", 0.6, ActionType.REVIEW, SeverityLevel.MEDIUM),
        ("toxic", 0.8, ActionType.WARN, SeverityLevel.MEDIUM),
        ("toxic", 0.5, ActionType.REVIEW, SeverityLevel.LOW),
        ("offensive", 0.7, ActionType.REVIEW, SeverityLevel.LOW),
        ("ambiguous", 0.9, ActionType.REVIEW, SeverityLevel.LOW),
        ("neutral", 0.8, ActionType.ALLOW, SeverityLevel.NONE),
    ],
)
async def test_action_recommendation_logic(
    label, confidence, expected_action, expected_severity
):
    classification = ClassificationResult(
        label=ClassificationLabel(label),
        confidence=confidence,
        reasoning="Some reasoning",
    )
    recommender = ActionRecommender()
    result = await recommender._execute(classification)

    assert result["action"] == expected_action
    assert result["severity"] == expected_severity
    assert isinstance(result["reasoning"], str)
    assert len(result["reasoning"]) > 10


def test_get_simple_recommendation():
    recommender = ActionRecommender()

    cases = {
        "hate": "Escalate to human moderator and restrict account",
        "toxic": "Issue warning and monitor behavior",
        "offensive": "Flag for review",
        "ambiguous": "Send for manual review",
        "neutral": "No action needed",
    }

    for label, expected in cases.items():
        classification = ClassificationResult(
            label=ClassificationLabel(label), confidence=0.8, reasoning="Doesn't matter"
        )
        assert recommender.get_simple_recommendation(classification) == expected


def test_get_simple_recommendation_invalid_input_raises():
    recommender = ActionRecommender()

    with pytest.raises(AgentExecutionError):
        recommender.get_simple_recommendation(None)
