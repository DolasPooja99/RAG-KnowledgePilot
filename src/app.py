import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv
import anthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

load_dotenv()

CONNECTION_STRING = "postgresql://localhost/knowledgepilot"
COLLECTION_NAME = "documents"
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_uploaded_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Tag each chunk with the filename
    for chunk in chunks:
        chunk.metadata["source"] = uploaded_file.name

    embeddings = get_embeddings()
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )
    os.unlink(tmp_path)
    return len(chunks)

def get_retriever():
    store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=get_embeddings(),
    )
    return store.as_retriever(search_kwargs={"k": 9})

st.title("KnowledgePilot")
st.caption("Upload a document and ask questions about it")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file:
        if st.button("Ingest PDF"):
            with st.spinner("Processing..."):
                num_chunks = ingest_uploaded_pdf(uploaded_file)
            st.success(f"Done — {num_chunks} chunks stored")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"]):
                    st.caption(f"Chunk {i+1}: {source}")

question = st.chat_input("Ask a question about your document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    retriever = get_retriever()
    chunks = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in chunks])
    sources = [f"{doc.metadata.get('source', 'unknown')} — {doc.page_content[:80]}..." for doc in chunks]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=f"""You are a helpful assistant. Answer using only the context below.
If the answer is not in the context, say 'I don't have that information.'

Context:
{context}""",
        messages=[{"role": "user", "content": question}]
    )

    answer = response.content[0].text
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("Sources"):
            for i, source in enumerate(sources):
                st.caption(f"Chunk {i+1}: {source}")