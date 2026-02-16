import json
from typing import List, Dict, Any

from config import get_or_create_index
from embeddings import embed_text

def load_faqs(path: str = "billing_faqs.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_vectors(faqs: List[Dict[str, Any]]):
    vectors = []
    for faq in faqs:
        emb = embed_text(faq["question"])
        vectors.append({
            "id": faq["id"],
            "values": emb,
            "metadata": {
                "question": faq["question"],
                "answer": faq["answer"],
                "tags": ",".join(faq.get("tags", [])),
                "source_url": faq.get("source_url", ""),
                "last_updated": faq.get("last_updated", ""),
                "official": faq.get("official", True),
            },
        })
    return vectors

def ingest():
    index = get_or_create_index(dimension=768)
    faqs = load_faqs()
    vectors = build_vectors(faqs)
    index.upsert(vectors=vectors)
    print(f"Ingested {len(vectors)} billing FAQs into Pinecone index.")

if __name__ == "__main__":
    ingest()
