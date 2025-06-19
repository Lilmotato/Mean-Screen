import pytest
from app.agents.retriever import HybridRetriever
from app.models.schemas import ClassificationLabel, ClassificationResult
from app.utils.exceptions import RetrievalError


@pytest.mark.asyncio
async def test_execute_returns_top_policies(mocker):
    mock_search = mocker.patch("app.agents.retriever.search_policies")
    mock_search.return_value = [
        {
            "id": "1",
            "score": 0.65,
            "data": {
                "title": "Policy A",
                "content": "Hate speech and harassment are prohibited.",
                "provider": "Meta",
                "type": "community_guidelines",
            },
        },
        {
            "id": "2",
            "score": 0.50,
            "data": {
                "title": "Policy B",
                "content": "Toxic language is discouraged.",
                "provider": "Reddit",
                "type": "platform_policy",
            },
        },
        {
            "id": "3",
            "score": 0.70,
            "data": {
                "title": "Policy C",
                "content": "No offensive behavior allowed.",
                "provider": "Google",
                "type": "legal_framework",
            },
        },
    ]

    classification = ClassificationResult(
        label=ClassificationLabel.hate,
        confidence=0.95,
        reasoning="harassment, hate, discrimination",
    )

    retriever = HybridRetriever()
    result = await retriever._execute("I hate you", classification)

    assert result.query_used == "I hate you"
    assert result.total_candidates == 3
    assert len(result.policies) == 3
    for policy in result.policies:
        assert policy.relevance_score <= 1.0
        assert policy.title in {"Policy A", "Policy B", "Policy C"}
        assert policy.explanation.startswith("Matched")


@pytest.mark.asyncio
async def test_execute_returns_empty_if_no_matches(mocker):
    mocker.patch("app.agents.retriever.search_policies", return_value=[])

    classification = ClassificationResult(
        label=ClassificationLabel.toxic, confidence=0.9, reasoning="abuse"
    )

    retriever = HybridRetriever()
    result = await retriever._execute("clean", classification)

    assert result.policies == []
    assert result.total_candidates == 0


@pytest.mark.asyncio
async def test_execute_invalid_input_raises_error():
    retriever = HybridRetriever()
    classification = ClassificationResult(
        label=ClassificationLabel.neutral, confidence=1.0, reasoning="no issues"
    )

    with pytest.raises(RetrievalError):
        await retriever._execute("", classification)


def test_score_and_explain_keywords_boost():
    retriever = HybridRetriever()
    classification = ClassificationResult(
        label=ClassificationLabel.hate,
        confidence=0.9,
        reasoning="harassment, discrimination and hate",
    )

    sample_result = [
        {
            "id": "p1",
            "score": 0.5,
            "data": {
                "title": "Policy X",
                "content": "Hate and harassment not allowed.",
                "provider": "Meta",
                "type": "community_guidelines",
            },
        }
    ]

    scored = retriever._score_and_explain(
        sample_result, "hateful message", classification
    )
    assert scored[0]["score"] > 0.5
    assert "Matched" in scored[0]["explanation"]
