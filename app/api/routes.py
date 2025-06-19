from fastapi import APIRouter, HTTPException

from app.agents.orchestrator import HateSpeechOrchestrator
from app.models.schemas import AnalyzeRequest, DetailedAnalyzeResponse

router = APIRouter(prefix="/api/v1", tags=["Hate Speech Detection"])
orchestrator = HateSpeechOrchestrator()


@router.post("/analyze", response_model=DetailedAnalyzeResponse)
async def analyze_text(payload: AnalyzeRequest):
    """
    Endpoint to analyze user text for hate speech detection, retrieve matching policies,
    generate reasoning, and recommend a moderation action.
    """
    try:
        return await orchestrator.run(payload.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing input: {str(e)}")
