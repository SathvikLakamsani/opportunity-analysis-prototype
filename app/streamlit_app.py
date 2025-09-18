import streamlit as st
import os

from src.utils.config import INDEX_PATH, KEYWORDS_PATH
from src.retrievers.serp_client import fetch_brand_snippets
from src.nlp.matcher import summarize_brand, top_k_keywords_from_summary

st.set_page_config(page_title="Opportunity Analysis Prototype", layout="centered")

st.title("🧭 Opportunity Analysis (Prototype)")
st.caption("Enter a brand or product name. The app will fetch context (via SerpAPI), summarize with OpenAI, and match to Walmart keywords via FAISS.")

with st.form("brand_form"):
    query = st.text_input("Brand/Product (e.g., 'Peigon sandals' or 'PawPure dog treats')", "")
    k = st.slider("Top K Keywords", 1, 10, 3, 1)
    submitted = st.form_submit_button("Analyze")

index_ok = os.path.exists(INDEX_PATH) and os.path.exists(KEYWORDS_PATH)
if not index_ok:
    st.warning("FAISS index not found. Run: `python scripts/build_index.py --csv data/walmart_keywords_sample.csv --out_dir .`")
    st.stop()

if submitted and query.strip():
    with st.spinner("Fetching context from the web..."):
        snippets = fetch_brand_snippets(query, num_results=5)

    st.subheader("Context Snippets")
    for s in snippets:
        st.write(f"- {s}")

    with st.spinner("Summarizing brand..."):
        summary = summarize_brand(snippets)
    st.subheader("Summary")
    st.write(summary)

    with st.spinner("Matching to Walmart keywords..."):
        results = top_k_keywords_from_summary(summary, k=k)

    st.subheader("Top Matches")
    for kw, score in results:
        st.write(f"**{kw}** — similarity: `{score:.3f}`")
