import os
import streamlit as st

from src.utils.config import INDEX_PATH, KEYWORDS_PATH
from src.retrievers.serp_client import fetch_brand_snippets
from src.nlp.matcher import summarize_brand, top_k_keywords_from_summary

st.set_page_config(page_title="Opportunity Analysis", layout="centered")
st.title("🧭 Opportunity Analysis (Prototype)")
st.caption(
    "Enter a brand or product; the app retrieves context (if available), "
    "summarizes, embeds, and matches to your keyword index."
)


@st.cache_resource(show_spinner=False)
def _index_exists() -> bool:
    return os.path.exists(INDEX_PATH) and os.path.exists(KEYWORDS_PATH)


with st.form("brand_form"):
    query = st.text_input("Brand/Product", "")
    k = st.slider("Top K Keywords", 1, 10, 3, 1)
    submitted = st.form_submit_button("Analyze")

if not _index_exists():
    st.warning(
        "FAISS index not found. Run: "
        "`python scripts/build_index.py --csv data/sample_keywords.csv --out_dir .`"
    )
elif submitted and query.strip():
    with st.spinner("Fetching context..."):
        snippets = fetch_brand_snippets(query, num_results=5)

    if snippets == [query]:
        st.info("Using your input directly (search API unavailable or unauthorized).")

    st.subheader("Context Snippets")
    for s in snippets:
        st.write(f"- {s}")

    with st.spinner("Summarizing..."):
        summary = summarize_brand(snippets)
    st.subheader("Summary")
    st.write(summary)

    with st.spinner("Matching keywords..."):
        results = top_k_keywords_from_summary(summary, k=k)

    st.subheader("Top Matches")
    for kw, score in results:
        st.write(f"**{kw}** — similarity: `{score:.3f}`")
