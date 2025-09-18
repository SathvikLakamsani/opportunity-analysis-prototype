import numpy as np
import faiss
from typing import Tuple, List
from src.utils.config import INDEX_PATH, KEYWORDS_PATH

def load_index() -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    index = faiss.read_index(INDEX_PATH)
    keywords = np.load(KEYWORDS_PATH, allow_pickle=True)
    return index, keywords

def search(index, keywords: np.ndarray, query_vec: np.ndarray, k: int = 3):
    D, I = index.search(query_vec, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        kw = str(keywords[idx])
        results.append((kw, float(score)))
    return results
