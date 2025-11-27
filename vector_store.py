# storage/vector_store.py
from __future__ import annotations

def is_available() -> bool:
    """Return True if vector dependencies are importable."""
    try:
        import chromadb  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return True
    except Exception:
        return False

def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return a sentence-transformers model or raise a clear error."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(
            "Vector store unavailable: install sentence-transformers (and chromadb/faiss-cpu)."
        ) from e

def get_client():
    """Return a ChromaDB client (lazy import)."""
    try:
        import chromadb
        return chromadb.Client()
    except Exception as e:
        raise RuntimeError(
            "Vector store unavailable: install chromadb."
        ) from e
