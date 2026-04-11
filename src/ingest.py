import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

load_dotenv()

CONNECTION_STRING = "postgresql://localhost/knowledgepilot"
COLLECTION_NAME = "documents"

def ingest_pdf(pdf_path: str):
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("Storing in pgvector...")
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )
    print("Done.")

if __name__ == "__main__":
    ingest_pdf("docs/pooja_resume_ggl.pdf")