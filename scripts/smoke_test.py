#!/usr/bin/env python3
"""
Quick smoke test to verify embeddings + FAISS retrieval works with the sample data.
- Assumes you've already run:
    python scripts/build_index.py --csv data/walmart_keywords_sample.csv --out_dir .
- Requires OPENAI_API_KEY in your environment (.env if using python-dotenv).
"""
import os
import numpy as np
from src.utils.config import OPENAI_API_KEY
from src.nlp.embed import get_embeddings
from src.nlp.index import load_index, search

def run_example(summary: str, k: int = 3):
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY not set. Put it in your .env or env vars.")
    vec = np.array(get_embeddings([summary]), dtype="float32")
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    index, keywords = load_index()
    results = search(index, keywords, vec, k=k)
    print(f"\nSummary: {summary}")
    for kw, score in results:
        print(f"  - {kw}  (sim={score:.3f})")

if __name__ == "__main__":
    print("✅ Smoke test running...")
    run_example("Brand sells women's walking sandals and comfy flip flops for travel.", k=3)
    run_example("Organic dog treats for puppies and large dogs; grain-free options.", k=3)
    run_example("Nonstick silicone baking mats and reusable silicone food bags.", k=3)
    print("\nDone.")
