# retrieval.py

import re
from typing import Dict, Any, List, Tuple

import ollama
from pinecone import Pinecone
from config import (
    CHAT_MODEL,
    EMBED_MODEL,
    get_or_create_index,
)

CONFIDENCE_THRESHOLD = 0.30  # similarity threshold for low confidence


def embed_text(text: str) -> List[float]:
    """
    Get an embedding from Ollama for the given text.
    """
    res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return res["embedding"]


def retrieve_chunks(query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Retrieve top_k chunks from Pinecone for the given query and return
    (matches, scores).
    """
    index = get_or_create_index(dimension=768)
    query_vec = embed_text(query)

    res = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )

    matches = res.get("matches", []) or []
    # Pinecone Python client typically returns matches as objects; support both dict/object
    chunks = []
    scores = []
    for m in matches:
        # m.score and m.metadata on newer clients, or dict-style on older ones
        score = getattr(m, "score", None) or m.get("score", 0.0)
        metadata = getattr(m, "metadata", None) or m.get("metadata", {})
        chunks.append(metadata)
        scores.append(score)

    return chunks, scores


def build_context_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Concatenate retrieved chunk texts into a single context string.
    """
    parts = []
    for i, ch in enumerate(chunks):
        text = ch.get("text", "")
        src = ch.get("source", "")
        parts.append(f"Chunk {i+1} (source: {src}):\n{text}\n")
    return "\n---\n".join(parts)


def call_llm(question: str, context: str) -> str:
    """
    Call Ollama chat model with retrieved context.
    """
    prompt = f"""You are a helpful T-Mobile billing FAQ assistant.
Use ONLY the provided context to answer the question concisely.
If the answer is not in the context, say you are not sure.

Context:
{context}

Question: {question}
Answer:"""

    res = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    # Adjust depending on Ollama's response schema
    return res["message"]["content"].strip()


def answer_billing_question(question: str) -> Dict[str, Any]:
    """
    Main entry point used by app.py.
    Returns dict: {"answer": str or None, "handoff": bool, "reason": str}
    """

    # 1) Detect explicit user request for human help
    wants_agent = bool(
        re.search(
            r"(human|agent|representative|real person|talk to someone|talk to a person|"
            r"contact someone|speak to someone|speak to an agent|customer service)",
            question,
            re.IGNORECASE,
        )
    )

    if wants_agent:
        return {
            "answer": "Okay, I’ll connect you to a human assistant.",
            "handoff": True,
            "reason": "user_requested",
        }

    # 2) Normal RAG flow: retrieve chunks from Pinecone
    chunks, scores = retrieve_chunks(question, top_k=5)
    best_score = scores[0] if scores else 0.0

    # 3) If low confidence: hand off instead of guessing
    if best_score < CONFIDENCE_THRESHOLD or not chunks:
        return {
            "answer": "I’m not fully sure of the answer from the billing FAQs.",
            "handoff": True,
            "reason": "low_confidence",
            "best_score": best_score,
        }

    # 4) Build context and ask the LLM
    context = build_context_from_chunks(chunks)
    answer = call_llm(question, context)

    # Optional: basic safeguard – if model says it's not sure, escalate
    if re.search(r"\b(not sure|cannot answer|do not know)\b", answer, re.IGNORECASE):
        return {
            "answer": answer,
            "handoff": True,
            "reason": "llm_unsure",
            "best_score": best_score,
        }

    return {
        "answer": answer,
        "handoff": False,
        "reason": "confident",
        "best_score": best_score,
    }
