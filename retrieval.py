from typing import List, Dict, Any

import ollama
from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBED_MODEL, CHAT_MODEL
from embeddings import embed_text

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def search_faqs(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    q_emb = embed_text(query)
    res = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True
    )
    # Pinecone Python client returns dict-like result; normalize to list of matches
    matches = res.get("matches", [])
    return matches

def build_context(matches: List[Dict[str, Any]]) -> str:
    parts = []
    for m in matches:
        md = m["metadata"]
        q = md.get("question", "")
        a = md.get("answer", "")
        src = md.get("source_url", "")
        parts.append(f"Q: {q}\nA: {a}\nSource: {src}")
    return "\n\n".join(parts)

def answer_billing_question(user_query: str) -> str:
    matches = search_faqs(user_query, top_k=3)
    if not matches or matches[0].get("score", 0.0) < 0.70:
        # Low confidence: push to official support
        return (
            "I’m not completely sure about this billing question. "
            "For accurate, account-specific information, please sign in to your T-Mobile account "
            "or contact T-Mobile Customer Care (for example, by dialing 611 from a T-Mobile phone)."
        )

    context = build_context(matches)

    system_prompt = (
        "You are a helpful assistant that answers ONLY T-Mobile billing and payment questions. "
        "Use the FAQ context below. If the answer is not clearly covered, say you are not sure "
        "and suggest contacting T-Mobile directly. "
        "Do not invent specific fees, dates, or policies.\n\n"
        f"FAQ context:\n{context}"
    )

    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    )
    return resp["message"]["content"]
