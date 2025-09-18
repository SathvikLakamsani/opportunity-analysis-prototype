from typing import List
from openai import OpenAI
from src.utils.config import OPENAI_API_KEY, EMBED_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    resp = _client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]
