import pytest
from app.agents.reasoner import PolicyReasoner
from app.models.schemas import PolicyDocument, ClassificationResult, ClassificationLabel
from app.utils.exceptions import AgentExecutionError


@pytest.mark.asyncio
async def test_execute_returns_explanation(mocker):
    mock_llm = mocker.Mock()
    mock_llm.reason_with_context = mocker.AsyncMock(
        return_value={
            "explanation": "Policies confirm this is hate speech.",
            "policy_summaries": {
                "p1": "Addresses hate content.",
                "p2": "Prohibits harassment.",
            },
        }
    )

    policies = [
        PolicyDocument(
            id="p1",
            title="Policy A",
            content="Hate speech and violence prohibited.",
            category="hate",
            relevance_score=0.9,
            source="Meta",
            policy_type="community_guidelines",
            explanation="Relevant to hate content.",
        ),
        PolicyDocument(
            id="p2",
            title="Policy B",
            content="Harassment is a violation.",
            category="abuse",
            relevance_score=0.8,
            source="Reddit",
            policy_type="platform_policy",
            explanation="Matches abuse scenario.",
        ),
    ]

    classification = ClassificationResult(
        label=ClassificationLabel.hate,
        confidence=0.92,
        reasoning="Detected hate language and threats.",
    )

    agent = PolicyReasoner(llm_service=mock_llm)
    result = await agent._execute(
        "I hate you and want you gone", policies, classification
    )

    assert "explanation" in result
    assert isinstance(result["policy_summaries"], dict)
    assert "p1" in result["policy_summaries"]
    assert "p2" in result["policy_summaries"]


@pytest.mark.asyncio
async def test_execute_invalid_llm_response_raises(mocker):
    mock_llm = mocker.Mock()
    mock_llm.reason_with_context = mocker.AsyncMock(return_value={"bad_key": "oops"})

    agent = PolicyReasoner(llm_service=mock_llm)

    with pytest.raises(
        AgentExecutionError, match="Failed to generate policy reasoning"
    ):
        await agent._execute(
            "text",
            [],
            ClassificationResult(
                label=ClassificationLabel.neutral,
                confidence=0.5,
                reasoning="Nothing problematic",
            ),
        )


@pytest.mark.asyncio
async def test_execute_llm_failure_raises(mocker):
    mock_llm = mocker.Mock()
    mock_llm.reason_with_context = mocker.AsyncMock(side_effect=Exception("timeout"))

    agent = PolicyReasoner(llm_service=mock_llm)

    with pytest.raises(AgentExecutionError):
        await agent._execute(
            "input",
            [],
            ClassificationResult(
                label=ClassificationLabel.toxic,
                confidence=0.9,
                reasoning="toxic language",
            ),
        )


def test_build_prompt_structure():
    agent = PolicyReasoner(llm_service=None)  # LLM not used in _build_prompt

    policies = [
        PolicyDocument(
            id="p99",
            title="Policy C",
            content="Example policy text that should appear.",
            category="hate",
            relevance_score=0.75,
            source="Google",
            policy_type="legal_framework",
            explanation="Relevant.",
        )
    ]

    classification = ClassificationResult(
        label=ClassificationLabel.hate,
        confidence=0.85,
        reasoning="discriminatory and hateful content",
    )

    prompt = agent._build_prompt("This is a test", policies, classification)

    assert "User Input" in prompt
    assert "Classification" in prompt
    assert "Relevant Policies" in prompt
    assert "Respond ONLY in this JSON format" in prompt
    assert "ID: p99" in prompt
    assert "Policy C" in prompt
