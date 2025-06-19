import pytest
from app.services.embed_service import get_embedding_service, EmbeddingService


def test_get_embedding_service_singleton():
    service1 = get_embedding_service()
    service2 = get_embedding_service()
    assert service1 is service2  # lru_cache ensures singleton


def test_embed_text_returns_list_of_floats():
    service = EmbeddingService()
    result = service.embed_text("test input")

    assert isinstance(result, list)
    assert all(isinstance(val, float) for val in result)
    assert len(result) == service.dimension


@pytest.mark.parametrize("bad_input", ["", None, 123])
def test_embed_text_invalid_inputs_raise(bad_input):
    service = EmbeddingService()
    with pytest.raises(ValueError):
        service.embed_text(bad_input)
