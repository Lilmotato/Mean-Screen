from fastapi import APIRouter, HTTPException
from app.models.policies import PolicyInput
from app.services.policy_store import store_policy
from app.services.qdrant_client import search_policies

router = APIRouter(prefix="/policy", tags=["Policy"])


@router.post("/add")
def add_policy(policy: PolicyInput):
    """Add a new policy"""
    try:
        policy_id = store_policy(policy)
        return {"message": "Policy stored successfully", "id": policy_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/search")
def search(query: str, limit: int = 3):
    """Search similar policies"""
    try:
        results = search_policies(query, limit)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))