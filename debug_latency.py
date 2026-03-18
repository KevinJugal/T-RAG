# debug_latency.py

import time
from retrieval import retrieve_chunks, build_context_from_chunks, call_llm

def debug_latency(question: str):
    print(f"Question: {question}\n")

    t0 = time.time()
    chunks, scores = retrieve_chunks(question)
    t1 = time.time()

    context = build_context_from_chunks(chunks)
    print("Context preview:\n", context[:800], "\n")
    t2 = time.time()
    answer = call_llm(question, context)
    t3 = time.time()

    print("Answer:\n", answer, "\n")
    print(f"Retrieval time: {t1 - t0:.2f} s")
    print(f"Context build time: {t2 - t1:.2f} s")
    print(f"Generation time: {t3 - t2:.2f} s")
    print(f"Total time: {t3 - t0:.2f} s")

if __name__ == "__main__":
    q = input("Enter a test question: ").strip()
    if q:
        debug_latency(q)
