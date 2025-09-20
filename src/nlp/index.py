import numpy as np
import faiss
from functools import lru_cache
from typing import List, Tuple

from src.utils.config import INDEX_PATH, KEYWORDS_PATH


@lru_cache(maxsize=1)
def load_index() -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    index = faiss.read_index(INDEX_PATH)
    keywords = np.load(KEYWORDS_PATH, allow_pickle=True)
    return index, keywords


def _normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def search(
    index: faiss.IndexFlatIP, keywords: np.ndarray, vecs: np.ndarray, k: int = 3
) -> List[Tuple[str, float]]:
    vecs = _normalize(vecs.astype("float32"))
    D, I = index.search(vecs, k)
    return [(str(keywords[j]), float(d)) for d, j in zip(D[0], I[0])]
