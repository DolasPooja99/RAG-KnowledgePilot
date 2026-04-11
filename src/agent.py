import os
from dotenv import load_dotenv
import anthropic
from retriever import get_retriever

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def ask(question: str):
    retriever = get_retriever()
    chunks = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in chunks])

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=f"""You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say 'I don't have that information.'

Context:
{context}""",
        messages=[
            {"role": "user", "content": question}
        ]
    )

    print(f"\nQ: {question}")
    print(f"A: {message.content[0].text}")

if __name__ == "__main__":
    ask("What is Pooja's work experience?")
    ask("What programming languages does Pooja know?")
    ask("What degree does Pooja have?")