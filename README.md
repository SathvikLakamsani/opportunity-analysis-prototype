
# Opportunity Analysis Prototype 

A minimal Streamlit + Python prototype for exploring how product or brand descriptions can be matched against a keyword database.

The app lets you:
- Enter a brand or product description
- (Optionally) fetch context snippets from a SERP provider
- Summarize the description with OpenAI
- Generate embeddings
- Retrieve the closest matching keywords from your dataset using FAISS
- Display the top results

---

## ⚠️ Data Notice
This project ships with a **sample keyword list** in `data/sample_keywords.csv` for demonstration purposes.

To use your own marketplace keyword data:
1. Prepare a CSV file with **one column named `keyword`**.
2. Save it in the `data/` folder, e.g. `data/my_keywords.csv`.
3. Rebuild the FAISS index:
   ```bash
   python scripts/build_index.py --csv data/my_keywords.csv --out_dir .
Restart the Streamlit app. The new keywords will be used for retrieval.

Setup
Prerequisites
Python 3.9–3.12

An OpenAI API key

(Optional) a SERP provider key:

Serpapi.com

or WebScrapingAPI’s Serp endpoint

Installation
bash
Copy code
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # then edit .env with your keys
In .env, set:

ini
Copy code
OPENAI_API_KEY=your_openai_key_here

# Choose a SERP provider (or leave blank to skip fetching snippets):
SERP_PROVIDER=webscrapingapi
WSA_SERP_API_KEY=your_webscrapingapi_key_here
SERPAPI_API_KEY=your_serpapi_key_here
Usage
1) Build the index (with sample or your own data)
bash
Copy code
python scripts/build_index.py --csv data/sample_keywords.csv --out_dir .
2) (Optional) Smoke test
bash
Copy code
PYTHONPATH=. python scripts/smoke_test.py
3) Run the app
bash
Copy code
PYTHONPATH=. streamlit run app/streamlit_app.py
Open the printed URL (usually http://localhost:8501).
Enter a product/brand description to see the top-matched keywords.

API Keys & Secrets
This project does not include working API keys.
You must provide your own keys in a local .env file.

.env is listed in .gitignore so your keys are never committed to git.

Everyone using this repo must set up their own API keys.

If deploying (e.g. Streamlit Cloud, Vercel), add your keys through the platform’s secrets manager.

Notes
If SERP API calls fail (401/429/etc.), the app falls back to using your input text directly.

You can swap in any marketplace keyword dataset by following the instructions above.

For production, consider using a persistent vector DB (like Chroma or Weaviate) instead of FAISS files.


