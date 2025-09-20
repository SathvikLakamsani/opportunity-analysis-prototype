import argparse
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

from src.nlp.embed import get_embeddings
from src.utils.config import EMBED_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with a 'keyword' column")
    ap.add_argument("--out_dir", default=".", help="Where to write index + keywords")
    ap.add_argument("--batch", type=int, default=2048, help="Embedding batch size")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=["keyword"])
    texts = df["keyword"].astype(str).tolist()

    embs = []
    for i in tqdm(range(0, len(texts), args.batch), desc="Embedding"):
        chunk = texts[i : i + args.batch]
        embs.extend(get_embeddings(chunk, model=EMBED_MODEL))

    X = np.array(embs, dtype="float32")

    # Normalize to compute cosine similarity via inner product
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, f"{args.out_dir}/embeddings.index")
    np.save(f"{args.out_dir}/keywords.npy", np.array(texts, dtype=object))
    print("✅ Wrote embeddings.index and keywords.npy")


if __name__ == "__main__":
    main()
