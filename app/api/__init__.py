
from fastapi import APIRouter
from app.api.routes import router as analyze_router
from app.api.policies import router as policy_router    

api_router = APIRouter()
api_router.include_router(analyze_router)
api_router.include_router(policy_router)  
