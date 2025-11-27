# storage/vector_store.py
from __future__ import annotations
"""
Local Vector Store helpers for CX Synthetic Insights PoC
- Chroma (persisted in data/chroma)
- SentenceTransformer embeddings (all-MiniLM-L6-v2)

pip install chromadb sentence-transformers
"""

import hashlib, os
from typing import Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", os.path.join("data", "chroma"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "cx_embeddings")

def chunk_text(text: str, size: int = 450, overlap: int = 80) -> List[str]:
    if not text:
        return []
    text = str(text)
    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks

def _stable_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()

def get_collection(name: Optional[str] = None):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    model_name = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    return client.get_or_create_collection(name or COLLECTION_NAME, embedding_function=embed_fn)

def upsert_embeddings(col,
                      df,
                      dataset_id: str,
                      text_col: str = "text",
                      batch_size: int = 256,
                      chunk_size: int = 450,
                      chunk_overlap: int = 80) -> int:
    required = {"interaction_id", text_col, "dataset_id"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"upsert_embeddings: missing columns: {missing}")

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, str]] = []
    added = 0

    for _, row in df.iterrows():
        iid = str(row.get("interaction_id"))
        text = row.get(text_col) or ""
        topic = str(row.get("topic", ""))
        stage = str(row.get("stage_guess", row.get("stage", "")))
        sentiment_label = str(row.get("sentiment_label", ""))
        source_pointer = f"events:{iid}"

        for idx, ch in enumerate(chunk_text(text, size=chunk_size, overlap=chunk_overlap)):
            emb_id = _stable_id(dataset_id, iid, str(idx))
            ids.append(emb_id)
            docs.append(ch)
            metas.append({
                "dataset_id": dataset_id,
                "interaction_id": iid,
                "chunk_index": str(idx),
                "topic": topic,
                "stage": stage,
                "sentiment_label": sentiment_label,
                "source_pointer": source_pointer,
            })

            if len(ids) >= batch_size:
                col.upsert(ids=ids, documents=docs, metadatas=metas)
                added += len(ids)
                ids, docs, metas = [], [], []

    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        added += len(ids)

    return added

def similarity_search(col, query: str, dataset_id: Optional[str] = None, k: int = 5):
    where = {"dataset_id": dataset_id} if dataset_id else None
    res = col.query(query_texts=[query], n_results=int(k), where=where)
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(ids)
    for i in range(len(ids)):
        out.append({
            "id": ids[i],
            "document": docs[i],
            "metadata": metas[i],
            "distance": dists[i],
        })
    return out
