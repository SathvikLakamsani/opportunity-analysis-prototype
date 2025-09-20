import numpy as np
from typing import List, Tuple
from openai import OpenAI

from src.utils.config import OPENAI_API_KEY, CHAT_MODEL
from src.nlp.embed import get_embeddings
from src.nlp.index import load_index, search

_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are a product analyst. Given web snippets about a brand/product, "
    "summarize what it sells and key attributes in 2-3 concise sentences."
)


def summarize_brand(snippets: List[str]) -> str:
    text = "\n".join(snippets[:5])[:4000]
    chat = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    return (chat.choices[0].message.content or "").strip()


def top_k_keywords_from_summary(summary: str, k: int = 3) -> List[Tuple[str, float]]:
    V = np.array(get_embeddings([summary]), dtype="float32")
    index, kw = load_index()
    return search(index, kw, V, k=k)
