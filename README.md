
# Opportunity Analysis Prototype (Optiwise)

A minimal Streamlit + Python prototype that:
- accepts a brand/product description,
- fetches external context via SerpAPI,
- summarizes with OpenAI,
- embeds with OpenAI embeddings,
- matches to the closest Walmart search keywords using a FAISS index,
- returns the top 3 matches.

> You can start with the included small sample keyword file and swap in the real Walmart keyword dump later.

---

## 1) Setup

```bash
# Clone your new repo or initialize locally
git init opportunity-analysis-prototype
cd opportunity-analysis-prototype

# (Optional but recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt
```

Create a `.env` by copying the example and filling in your keys:

```bash
cp .env.example .env
# edit .env to add OPENAI_API_KEY and SERPAPI_API_KEY
```

---

## 2) Prepare Keyword Index

A small sample file is included at `data/walmart_keywords_sample.csv`.
When you obtain the full Walmart keyword dataset, replace this file (or add a new file) with your full list.
The *only* required column is `keyword`.

Then build the FAISS index:

```bash
python scripts/build_index.py --csv data/walmart_keywords_sample.csv --out_dir .
```

This will create:
- `embeddings.index` (FAISS index file)
- `keywords.npy` (NumPy array of original keyword strings)

---

## 3) Run the App

```bash
streamlit run app/streamlit_app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

---

## 3.5) Smoke Test (optional)

After building the index, you can run a quick smoke test (no SerpAPI, no Chat — just embeddings + retrieval):

```bash
python scripts/smoke_test.py
```

You should see the top matches for a few example summaries printed to the console.

---

## 4) Repo Structure

```
.
├── app/
│   └── streamlit_app.py          # Streamlit UI and end-to-end flow
├── data/
│   └── walmart_keywords_sample.csv
├── scripts/
│   ├── build_index.py            # Build FAISS index from CSV of keywords
│   └── smoke_test.py             # Quick embeddings + FAISS sanity check
├── src/
│   ├── nlp/
│   │   ├── embed.py              # OpenAI embedding helpers
│   │   ├── index.py              # FAISS index load/query helpers
│   │   └── matcher.py            # Brand→summary→embedding→top-k keywords
│   ├── retrievers/
│   │   └── serp_client.py        # SerpAPI fetcher for brand context
│   └── utils/
│       └── config.py             # Env loading, shared config
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 5) First GitHub Push

```bash
git add .
git commit -m "Initial prototype scaffold"
git branch -M main
git remote add origin https://github.com/<your-username>/opportunity-analysis-prototype.git
git push -u origin main
```

---

## 6) Swapping in Real Data

1. Put your full Walmart keyword list into `data/your_full_keywords.csv` with a single column named `keyword`.
2. Rebuild the index:

```bash
python scripts/build_index.py --csv data/your_full_keywords.csv --out_dir .
```

3. Restart Streamlit.

---

## 7) Notes

- Keep the FAISS index and keyword array committed if they’re small; otherwise add to `.gitignore` and regenerate in CI.
- For larger datasets, consider chunking and using persistent vector DBs (Chroma, Weaviate). For Phase 1, FAISS files are simplest.
- If SerpAPI rate limits you, start with just your raw brand text and skip retrieval, then add retrieval later.
