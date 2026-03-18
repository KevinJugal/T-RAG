# app.py

from retrieval import answer_billing_question
from call_service import call_support  # this module wraps your telephony provider


def main():
    print("Voicemail / Billing FAQ RAG (Python + Ollama + Pinecone)")
    print("Type your question, or 'exit' to quit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[System] Exiting.")
            break

        if not q:
            continue

        if q.lower() in {"exit", "quit"}:
            print("[System] Goodbye.")
            break

        # Run your RAG billing QA pipeline
        # Expected shape:
        # {
        #   "answer": str,
        #   "handoff": bool,
        #   "reason": str,
        #   "best_score": float,
        #   ...
        # }
        result = answer_billing_question(q)
        answer = result.get("answer", "")
        handoff = result.get("handoff", False)
        reason = result.get("reason", "")
        best_score = result.get("best_score")

        # Debug line so you can compare behavior with debug_latency.py
        print(f"[DEBUG] result={result}")

        # Show AI answer (may be empty when handing off)
        if answer:
            print(f"\nAssistant: {answer}\n")

        # If handoff is requested, trigger human escalation
        if handoff:
            print("[System] A human assistant will take over this conversation.")
            if reason:
                print(f"[System] Handoff reason: {reason} (best_score={best_score})")
            print("[System] Initiating a call to the support number...\n")

            try:
                # If you collect a customer phone earlier, pass it here
                # e.g. call_support(customer_number="+1XXXXXXXXXX")
                call_support()
            except Exception as e:
                # Fallback if call API fails
                print(f"[System] Failed to initiate call: {e}\n")


if __name__ == "__main__":
    main()
