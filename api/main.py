from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import api_router
from app.services.qdrant_client import init_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_collection()
    yield


app = FastAPI(title="Hate Speech Policy API", version="1.0", lifespan=lifespan)

app.include_router(api_router)
