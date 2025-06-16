from app.models.policies import PolicyInput
from app.services.qdrant_client import add_policy
from app.services.embed_service import get_embedding_service

embedding_service = get_embedding_service()

def store_policy(policy: PolicyInput) -> str:
    """Process and store policy"""
    if not policy.text or not policy.provider or not policy.type:
        raise ValueError("Missing required fields")
    
    vector = embedding_service.embed_text(policy.text)
    metadata = {
        "provider": policy.provider,
        "type": policy.type,
        "text": policy.text
    }
    
    return add_policy(vector, metadata)