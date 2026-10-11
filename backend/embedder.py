"""
embedder.py — Cached, retry-safe embedding wrapper around Ollama nomic-embed-text.
LRUCache(2000) prevents re-embedding the same text across requests.
"""
import asyncio
import math
import os
from collections import OrderedDict
from typing import List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


class LRUCache:
    def __init__(self, capacity: int = 2000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[List[float]]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: List[float]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


embedding_cache = LRUCache(capacity=2000)
_embed_semaphore = asyncio.Semaphore(1)


async def embed_text(text: str, max_retries: int = 5) -> List[float]:
    """Embed text via Ollama with LRU cache and exponential backoff."""
    if not text or not text.strip():
        return []
    cached = embedding_cache.get(text)
    if cached is not None:
        return cached
    async with _embed_semaphore:
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(
                    headers={"ngrok-skip-browser-warning": "true"}
                ) as client:
                    resp = await client.post(
                        f"{OLLAMA_BASE_URL}/api/embeddings",
                        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                    embedding = resp.json().get("embedding", [])
                    if embedding:
                        embedding_cache.put(text, embedding)
                    return embedding
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    wait = 15 * (attempt + 1)
                    print(f"[Embedder] Rate-limited (403). Waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))
                    else:
                        print(f"[Embedder] All retries failed: {text[:50]}...")
                        return []
            except Exception as e:
                print(f"[Embedder] Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    return []
    return []


async def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts sequentially using the cache."""
    return [await embed_text(t) for t in texts]


def is_valid_embedding(emb) -> bool:
    """
    Verifies that an embedding is an accepted 1D numeric vector:
    1. Accepted 1D sequence (list, tuple, or numpy array).
    2. Non-empty.
    3. All values are numeric.
    4. Boolean values are strictly NOT accepted as numeric vector components.
    5. All values are finite (no NaN, inf, -inf).
    6. Vector norm is finite and strictly greater than zero.
    """
    if emb is None:
        return False
    if hasattr(emb, "ndim"):
        if emb.ndim != 1 or emb.size == 0:
            return False
        if hasattr(emb, "dtype") and emb.dtype == bool:
            return False
        emb_iterable = emb
    elif isinstance(emb, (list, tuple)):
        if len(emb) == 0:
            return False
        emb_iterable = emb
    else:
        return False

    norm_sq = 0.0
    for v in emb_iterable:
        if isinstance(v, bool):
            return False
        if not isinstance(v, (int, float)):
            try:
                if hasattr(v, "item"):
                    v = v.item()
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        return False
                else:
                    return False
            except Exception:
                return False
        if math.isnan(v) or math.isinf(v):
            return False
        norm_sq += float(v * v)

    if math.isnan(norm_sq) or math.isinf(norm_sq) or norm_sq <= 0.0:
        return False

    return True


def safe_cosine_similarity(a: List[float], b: List[float]) -> Optional[float]:
    """
    Defensively calculates cosine similarity between two validated vectors.
    Returns float in [-1.0, 1.0] if finite, else None.
    """
    try:
        if not is_valid_embedding(a) or not is_valid_embedding(b):
            return None
        if len(a) != len(b):
            return None
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        mag_a = math.sqrt(sum(float(x) * float(x) for x in a))
        mag_b = math.sqrt(sum(float(y) * float(y) for y in b))
        if mag_a <= 0.0 or mag_b <= 0.0:
            return None
        sim = dot / (mag_a * mag_b)
        if math.isnan(sim) or math.isinf(sim):
            return None
        return sim
    except Exception as e:
        print(f"[Embedding] Error during cosine similarity calculation: {e}")
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Safe cosine similarity between two vectors, returning 0.0 on failure."""
    sim = safe_cosine_similarity(a, b)
    return sim if sim is not None else 0.0
