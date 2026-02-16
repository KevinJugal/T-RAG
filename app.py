from retrieval import answer_billing_question

def main():
    print("T-Mobile Billing FAQ RAG (Python + Ollama + Pinecone)")
    print("Type your billing question, or 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        answer = answer_billing_question(q)
        print(f"\nAssistant: {answer}\n")

if __name__ == "__main__":
    main()
