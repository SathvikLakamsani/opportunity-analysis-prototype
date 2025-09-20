import time
import requests
from typing import List, Dict, Any

from src.utils.config import SERP_PROVIDER, provider_key


def _parse_snippets(payload: Dict[str, Any], k: int) -> List[str]:
    out: List[str] = []
    for it in (payload.get("organic_results") or [])[:k]:
        s = it.get("snippet") or it.get("title") or ""
        if s:
            out.append(s)
    return out


def fetch_brand_snippets(query: str, num_results: int = 5) -> List[str]:
    """
    Fetch web context using either:
      - serpapi.com (SERP_PROVIDER=serpapi)
      - WebScrapingAPI Serp endpoint (SERP_PROVIDER=webscrapingapi)
    Returns [query] on any error/unauthorized so the app never crashes.
    """
    k = max(1, min(num_results, 10))
    key = provider_key()
    if not key:
        return [query]

    base = (
        "https://serpapi.com/search.json"
        if SERP_PROVIDER == "serpapi"
        else "https://serpapi.webscrapingapi.com/v2"
    )
    params = {"engine": "google", "q": query, "num": k, "api_key": key}
    headers = {"User-Agent": "OppAnalysis/1.0"}

    for attempt in range(3):
        try:
            r = requests.get(base, params=params, headers=headers, timeout=20)
            if r.status_code in (401, 402, 403, 429):  # unauthorized/payment/forbidden/ratelimit
                return [query]
            r.raise_for_status()
            data = r.json()
            snips = _parse_snippets(data, k)
            return snips or [query]
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            return [query]
