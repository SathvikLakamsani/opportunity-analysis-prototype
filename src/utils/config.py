import os
from dotenv import load_dotenv
load_dotenv()

SERP_PROVIDER = os.getenv("SERP_PROVIDER", "serpapi").lower().strip()

# Official serpapi.com key (if you use that)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")

# WebScrapingAPI's Serp endpoint key
WSA_SERP_API_KEY = os.getenv("WSA_SERP_API_KEY", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
INDEX_PATH = os.getenv("INDEX_PATH", "embeddings.index")
KEYWORDS_PATH = os.getenv("KEYWORDS_PATH", "keywords.npy")
