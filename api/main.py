# main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api import api_router
from app.services.qdrant_client import init_collection

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize vector DB collection (Qdrant)
    init_collection()
    yield

app = FastAPI(
    title="Hate Speech Policy API",
    version="1.0",
    lifespan=lifespan
)

app.include_router(api_router)
