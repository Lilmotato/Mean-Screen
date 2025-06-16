# app/api/v1/routes.py
from fastapi import APIRouter, HTTPException
from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.agents.orchestrator import HateSpeechOrchestrator

router = APIRouter(prefix="/api/v1", tags=["Hate Speech Detection"])
orchestrator = HateSpeechOrchestrator()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(payload: AnalyzeRequest):
    try:
        return await orchestrator.run(payload.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
