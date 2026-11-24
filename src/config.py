import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "development")
STORAGE_MODE = os.getenv("STORAGE_MODE", "local").strip().lower()
if STORAGE_MODE not in {"local", "cloud"}:
    raise ValueError("STORAGE_MODE must be local or cloud")

# Optional future cloud storage. Local mode ignores these credentials.
SUPABASE_DB_URL = (os.getenv('SUPABASE_DB_URL') or os.getenv('supabase_db_url')) if STORAGE_MODE == "cloud" else None
    

# Optional cloud vector search; local mode searches the bundled documents.
QDRANT_URL = os.getenv("QDRANT_URL") if STORAGE_MODE == "cloud" else None
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") if STORAGE_MODE == "cloud" else None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "nia_gemini_chunks")

# ── Chunking (Unchanged) ────────────────────────────────────────────────────
PARENT_CHUNK_SIZE = 1000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50

# ── Retrieval ───────────────────────────────────────────────────────────────
RETRIEVER_K = 5

# ── Embedding ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIMENSIONS = 3072

# ── LangGraph Limits ────────────────────────────────────────────────────────
MAX_RETRIES = 3
MAX_REWRITE_TRIES = 3
