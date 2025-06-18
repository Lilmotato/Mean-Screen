from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from app.services.embed_service import get_embedding_service
import os
import uuid

client = QdrantClient(url=os.getenv("QDRANT_HOST", "http://localhost:6333"))
COLLECTION_NAME = "policies"


def init_collection():
    """Initialize collection if it doesn't exist"""
    embedding_service = get_embedding_service()
    collections = [col.name for col in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_service.dimension,
                distance=Distance.COSINE
            )
        )

def add_policy(vector: list[float], metadata: dict) -> str:
    """Add policy to Qdrant"""
    embedding_service = get_embedding_service()
    if not isinstance(vector, list) or len(vector) != embedding_service.dimension:
        raise ValueError("Invalid vector format")
    
    policy_id = str(uuid.uuid4())
    point = PointStruct(id=policy_id, vector=vector, payload=metadata)
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    return policy_id


def search_policies(query: str, limit: int = 3) -> list:
    """Search for similar policies"""
    embedding_service = get_embedding_service()
    query_vector = embedding_service.embed_text(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )
    return [{"id": r.id, "score": r.score, "data": r.payload} for r in results]