from typing import List
from time import sleep
from openai import OpenAI, RateLimitError

from src.utils.config import OPENAI_API_KEY, EMBED_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)


def get_embeddings(
    texts: List[str], model: str = EMBED_MODEL, batch_size: int = 256
) -> List[List[float]]:
    """
    Batched embeddings with simple exponential backoff on rate limits.
    Falls back to zero vectors for failed chunks to keep pipeline moving.
    """
    embs: List[List[float]] = []
    i = 0
    # Default vector size for text-embedding-3-small is 1536; adjust if you switch models
    fallback_dim = 1536
    while i < len(texts):
        chunk = texts[i : i + batch_size]
        for attempt in range(3):
            try:
                resp = _client.embeddings.create(model=model, input=chunk)
                embs.extend([d.embedding for d in resp.data])
                break
            except RateLimitError:
                sleep(1.5 * (attempt + 1))
            except Exception:
                embs.extend([[0.0] * fallback_dim] * len(chunk))
                break
        i += batch_size
    return embs
