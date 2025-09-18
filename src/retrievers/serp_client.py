import requests
from typing import List, Dict, Any
from src.utils.config import SERP_PROVIDER, SERPAPI_API_KEY, WSA_SERP_API_KEY

def _parse_snippets(payload: Dict[str, Any], num_results: int) -> List[str]:
    """Extract snippet/title text from a SerpAPI-like JSON."""
    results = []
    items = (payload.get("organic_results") or [])[:num_results]
    for it in items:
        snip = it.get("snippet") or it.get("title") or ""
        if snip:
            results.append(snip)
    return results

def fetch_brand_snippets(query: str, num_results: int = 5) -> List[str]:
    """
    Fetch web context using either:
      - official serpapi.com  (SERP_PROVIDER=serpapi)
      - WebScrapingAPI Serp endpoint (SERP_PROVIDER=webscrapingapi)
    If anything fails (401, rate limit, network), gracefully fall back to [query].
    """
    num = max(1, min(num_results, 10))

    try:
        if SERP_PROVIDER == "webscrapingapi":
            api_key = WSA_SERP_API_KEY
            if not api_key:
                return [query]

            # WebScrapingAPI's Serp endpoint (SerpAPI-compatible)
            base_url = "https://serpapi.webscrapingapi.com/v2"
            params = {
                "engine": "google",
                "q": query,
                "num": num,
                "api_key": api_key,
            }
            r = requests.get(base_url, params=params, timeout=20)
            if r.status_code in (401, 402, 403, 429):
                return [query]
            r.raise_for_status()
            data = r.json()
            snippets = _parse_snippets(data, num)
            return snippets or [query]

        else:
            # Default: official serpapi.com
            api_key = SERPAPI_API_KEY
            if not api_key:
                return [query]

            base_url = "https://serpapi.com/search.json"
            params = {
                "engine": "google",
                "q": query,
                "num": num,
                "api_key": api_key,
            }
            r = requests.get(base_url, params=params, timeout=20)
            if r.status_code in (401, 402, 403, 429):
                return [query]
            r.raise_for_status()
            data = r.json()
            snippets = _parse_snippets(data, num)
            return snippets or [query]

    except Exception:
        # Any error → don't crash; use the input as context
        return [query]
