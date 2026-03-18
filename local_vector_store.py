# local_vector_store.py

import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tmobile_billing_faq")


def get_chroma_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(is_persistent=True)
    )
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def upsert_vectors(vectors: List[Dict[str, Any]]):
    """
    vectors: list of {
      "id": str,
      "values": List[float],
      "metadata": { "text": str, "source": str, ... }
    }
    """
    collection = get_chroma_collection()

    ids = [v["id"] for v in vectors]
    embeddings = [v["values"] for v in vectors]
    metadatas = [v.get("metadata", {}) for v in vectors]

    # text is stored inside metadata["text"], so documents can be empty or duplicate
    documents = [m.get("text", "") for m in metadatas]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query_vectors(query_embedding: List[float], top_k: int = 3):
    """
    Returns (metadatas, distances).
    """
    collection = get_chroma_collection()

    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    return metadatas, distances
