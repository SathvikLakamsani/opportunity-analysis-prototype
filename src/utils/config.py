import os
from functools import lru_cache
from dotenv import load_dotenv

# Load local .env if present; platforms can also inject env vars directly
load_dotenv()

SERP_PROVIDER = os.getenv("SERP_PROVIDER", "webscrapingapi").lower().strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Official serpapi.com
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
# WebScrapingAPI's Serp endpoint (serpapi-compatible)
WSA_SERP_API_KEY = os.getenv("WSA_SERP_API_KEY", "")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
INDEX_PATH = os.getenv("INDEX_PATH", "embeddings.index")
KEYWORDS_PATH = os.getenv("KEYWORDS_PATH", "keywords.npy")


@lru_cache(maxsize=1)
def provider_key() -> str:
    """Return the active SERP provider key based on SERP_PROVIDER."""
    if SERP_PROVIDER == "serpapi":
        return SERPAPI_API_KEY
    return WSA_SERP_API_KEY


def have_openai() -> bool:
    return bool(OPENAI_API_KEY)


def have_serp() -> bool:
    return bool(provider_key())
