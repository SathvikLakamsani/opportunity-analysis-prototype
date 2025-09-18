import numpy as np
from typing import List, Tuple
from openai import OpenAI
from src.utils.config import OPENAI_API_KEY, CHAT_MODEL
from src.nlp.embed import get_embeddings
from src.nlp.index import load_index, search

_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are a product analyst. Given raw web snippets about a brand/product, "
    "summarize what it sells and key attributes in 2-3 concise sentences."
)

def summarize_brand(snippets: List[str]) -> str:
    msg = "\n".join(snippets[:5])
    chat = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg},
        ],
        temperature=0.2,
    )
    return chat.choices[0].message.content.strip()

def top_k_keywords_from_summary(summary: str, k: int = 3):
    vec = np.array(get_embeddings([summary]), dtype="float32")
    faiss_vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    index, keywords = load_index()
    return search(index, keywords, faiss_vec, k=k)
