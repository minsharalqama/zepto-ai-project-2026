from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents() -> tuple[list[str], list[str], list[dict]]:
    documents = []
    ids = []
    metadatas = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        doc_id = path.stem
        ids.append(doc_id)
        documents.append(text)
        metadatas.append({"source": doc_id, "filename": path.name})
    if len(documents) != 8:
        raise RuntimeError(f"Expected 8 corpus documents, found {len(documents)}")
    return documents, ids, metadatas


def build_index() -> None:
    documents, ids, metadatas = load_documents()
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )
    existing = collection.get(include=[])
    existing_ids = set(existing.get("ids", []))
    if existing_ids:
        collection.delete(ids=list(existing_ids))

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"Indexed {len(ids)} document chunks in ChromaDB collection '{COLLECTION_NAME}'.")
    print(f"Persistent directory: {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
