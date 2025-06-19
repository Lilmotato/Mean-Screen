from functools import lru_cache

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Centralized embedding service"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
