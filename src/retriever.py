import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

load_dotenv()

CONNECTION_STRING = "postgresql://localhost/knowledgepilot"
COLLECTION_NAME = "documents"

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )
    return store.as_retriever(search_kwargs={"k": 3})

def retrieve(question: str):
    retriever = get_retriever()
    results = retriever.invoke(question)
    for i, doc in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(doc.page_content)

if __name__ == "__main__":
    retrieve("What is Pooja's work experience?")