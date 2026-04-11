import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    system="You are a helpful assistant that answers questions based only on the context provided. If the answer is not in the context, say so.",
    messages=[
        {
            "role": "user",
            "content": "What is RAG in AI? Explain in 3 sentences."
        }
    ]
)

print(message.content[0].text)
print(f"\nTokens used: {message.usage.input_tokens} in, {message.usage.output_tokens} out")