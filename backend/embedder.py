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


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
