import argparse
import numpy as np
import pandas as pd
import faiss
from src.nlp.embed import get_embeddings
from src.utils.config import EMBED_MODEL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with column 'keyword'")
    ap.add_argument("--out_dir", default=".", help="Where to write index + keywords")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "keyword" not in df.columns:
        raise ValueError("CSV must have a 'keyword' column")

    texts = df["keyword"].astype(str).tolist()
    embs = get_embeddings(texts, model=EMBED_MODEL)
    X = np.array(embs, dtype="float32")
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, f"{args.out_dir}/embeddings.index")
    np.save(f"{args.out_dir}/keywords.npy", np.array(texts, dtype=object))
    print("Wrote embeddings.index and keywords.npy")

if __name__ == "__main__":
    main()
