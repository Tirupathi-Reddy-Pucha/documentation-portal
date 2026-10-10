import pytest
import asyncio
from embedder import embed_text, cosine_similarity

@pytest.mark.asyncio
async def test_embedder_cache():
    # Cache miss
    emb1 = await embed_text("Hello World")
    assert isinstance(emb1, list)
    assert len(emb1) == 768

    # Cache hit
    emb2 = await embed_text("Hello World")
    assert emb1 is emb2  # Exact same object from cache

def test_cosine_similarity():
    # Identical vectors
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert cosine_similarity(v1, v2) == 1.0

    # Orthogonal vectors
    v3 = [0.0, 1.0, 0.0]
    assert cosine_similarity(v1, v3) == 0.0

    # Opposite vectors
    v4 = [-1.0, 0.0, 0.0]
    assert cosine_similarity(v1, v4) == -1.0
