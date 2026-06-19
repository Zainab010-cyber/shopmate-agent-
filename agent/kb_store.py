"""ChromaDB knowledge base for Campus Shop FAQ articles."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "data" / "kb"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "campus_shop_kb"

_client: chromadb.PersistentClient | None = None
_collection = None


def _get_embedding_function():
    """Use Chroma default embeddings for the local FAQ index.

    Keeps the KB reproducible without extra OpenAI calls and avoids conflicts
    when OPENAI_API_KEY is added after the index was first built.
    """
    return embedding_functions.DefaultEmbeddingFunction()


def _create_collection(embedding_function):
    """Create a fresh collection with the given embedding function."""
    global _collection
    _collection = _client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def get_collection():
    """Return or create the ChromaDB collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_function = _get_embedding_function()

    try:
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
    except ValueError as exc:
        if "Embedding function conflict" not in str(exc):
            raise
        # Rebuild when an older index used a different embedding backend.
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        _collection = None
        _create_collection(embedding_function)

    return _collection


def _load_kb_documents() -> list[tuple[str, str, dict]]:
    """Load markdown KB files as (id, text, metadata) tuples."""
    documents: list[tuple[str, str, dict]] = []
    if not KB_DIR.exists():
        return documents

    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_id = path.stem
        title_line = next((line for line in text.splitlines() if line.startswith("# ")), f"# {doc_id}")
        title = title_line.lstrip("# ").strip()
        documents.append((doc_id, text, {"title": title, "source": path.name}))
    return documents


def init_knowledge_base(force_rebuild: bool = False) -> int:
    """Embed KB articles into ChromaDB. Returns document count."""
    global _client, _collection
    if force_rebuild:
        if _client is not None:
            try:
                _client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
        _collection = None
    collection = get_collection()

    documents = _load_kb_documents()
    if not documents:
        return 0

    if force_rebuild or collection.count() == 0:
        existing_ids: set[str] = set()
    else:
        existing_ids = set(collection.get()["ids"])
    new_docs = [(i, t, m) for i, t, m in documents if i not in existing_ids]

    if new_docs:
        collection.add(
            ids=[d[0] for d in new_docs],
            documents=[d[1] for d in new_docs],
            metadatas=[d[2] for d in new_docs],
        )

    return collection.count()


def search_knowledge_base(query: str, n_results: int = 3) -> list[dict]:
    """Semantic search over FAQ articles."""
    collection = get_collection()
    if collection.count() == 0:
        init_knowledge_base()

    results = collection.query(query_texts=[query], n_results=min(n_results, collection.count() or 1))
    hits: list[dict] = []
    if not results["documents"] or not results["documents"][0]:
        return hits

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - dist
        hits.append(
            {
                "title": meta.get("title", "Unknown"),
                "source": meta.get("source", ""),
                "content": doc,
                "similarity": round(similarity, 3),
            }
        )
    return hits
