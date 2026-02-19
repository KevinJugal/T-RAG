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
        txt = md.get("text", "")
        src = md.get("source", "")
        parts.append(f"{txt}\n(Source: {src})")
    return "\n\n---\n\n".join(parts)


def answer_billing_question(user_query: str) -> str:
    matches = search_faqs(user_query, top_k=5)

    # If nothing at all is found, then fall back
    if not matches:
        return (
            "I’m not completely sure about this question. "
            "Please contact support for more help."
        )

    context = build_context(matches)
    system_prompt = (
        "You are a helpful assistant that answers questions about voicemail and visual voicemail "
        "for Telecom customers. Use the context below. "
        "If the answer is not clearly covered, say you are not sure and suggest contacting support.\n\n"
        f"Context:\n{context}"
    )

    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    )
    return resp["message"]["content"]

