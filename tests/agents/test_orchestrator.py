import pytest
from app.agents.orchestrator import HateSpeechOrchestrator
from app.models.schemas import (
    ClassificationLabel,
    ClassificationResult,
    PolicyDocument,
    ActionRecommendation,
    DetailedAnalyzeResponse,
    ActionType,
    SeverityLevel,
)


@pytest.mark.asyncio
async def test_run_returns_detailed_response(mocker):
    orchestrator = HateSpeechOrchestrator()

    mocker.patch.object(
        orchestrator.error_handler, "validate_input", return_value={"valid": True}
    )

    mocker.patch.object(
        orchestrator.detector,
        "_execute",
        return_value=ClassificationResult(
            label=ClassificationLabel.hate,
            confidence=0.91,
            reasoning="Contains hate and threats",
        ),
    )

    mock_policies = [
        PolicyDocument(
            id="p1",
            title="Policy A",
            content="No hate speech allowed.",
            category="hate",
            relevance_score=0.85,
            source="Meta",
            policy_type="community_guidelines",
            explanation="",
        )
    ]
    mocker.patch.object(
        orchestrator.retriever,
        "_execute",
        return_value=mocker.Mock(policies=mock_policies),
    )

    mocker.patch.object(
        orchestrator.reasoner,
        "_execute",
        return_value={
            "explanation": "Policy A supports the hate classification.",
            "policy_summaries": {"p1": "This explicitly prohibits hate speech."},
        },
    )

    mocker.patch.object(
        orchestrator.recommender,
        "_execute",
        return_value=ActionRecommendation(
            action=ActionType.ESCALATE,
            severity=SeverityLevel.HIGH,
            reasoning="High confidence hate speech",
        ),
    )

    result: DetailedAnalyzeResponse = await orchestrator.run(
        "You should be eliminated."
    )

    assert result.hate_speech.classification == "Hate"
    assert result.hate_speech.confidence.value == "High"
    assert "hate" in result.hate_speech.reason.lower()
    assert len(result.policies) == 1
    assert "Policy A" in result.policies[0].summary
    assert "This explicitly prohibits hate speech" in result.policies[0].summary
    assert result.reasoning == "Policy A supports the hate classification."
    assert result.action.action == ActionType.ESCALATE


@pytest.mark.asyncio
async def test_run_invalid_input_raises(mocker):
    orchestrator = HateSpeechOrchestrator()
    mocker.patch.object(
        orchestrator.error_handler,
        "validate_input",
        return_value={"valid": False, "message": "Input is empty"},
    )

    with pytest.raises(ValueError, match="Input is empty"):
        await orchestrator.run("")


@pytest.mark.asyncio
async def test_run_handles_internal_failure(mocker):
    orchestrator = HateSpeechOrchestrator()
    mocker.patch.object(
        orchestrator.error_handler, "validate_input", return_value={"valid": True}
    )
    mocker.patch.object(
        orchestrator.detector, "_execute", side_effect=RuntimeError("LLM crash")
    )
    mocker.patch.object(
        orchestrator.error_handler,
        "handle_error",
        return_value={"error": "LLM crash"},
    )

    with pytest.raises(RuntimeError, match="LLM crash"):
        await orchestrator.run("I hate everyone")
