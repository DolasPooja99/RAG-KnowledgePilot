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
from flashcards import detect_book_and_chapters, generate_flashcards

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

    is_book, chapters = detect_book_and_chapters(pages)
    return len(chunks), pages, is_book, chapters

def get_retriever():
    store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=get_embeddings(),
    )
    return store.as_retriever(search_kwargs={"k": 9})

# ── Page setup ────────────────────────────────────────────────────────────────
st.title("RAG-KnowledgePilot")
st.caption("Upload a document and ask questions about it")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "book_files" not in st.session_state:
    st.session_state.book_files = {}   # filename -> {pages, chapters}
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []   # [{question, answer, chapter}]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file:
        if st.button("Ingest PDF"):
            with st.spinner("Processing..."):
                num_chunks, pages, is_book, chapters = ingest_uploaded_pdf(uploaded_file)
            st.success(f"Done — {num_chunks} chunks stored")

            if is_book:
                st.info(f"Book detected — {len(chapters)} chapters found")
                st.session_state.book_files[uploaded_file.name] = {
                    "pages": pages,
                    "chapters": chapters,
                }
            else:
                st.caption("Not a book — flashcard generation is only available for books.")

    # Flashcard generation (only shown when at least one book is ingested)
    if st.session_state.book_files:
        st.divider()
        st.header("Flashcards")

        book_name = st.selectbox("Book", list(st.session_state.book_files.keys()))
        book_data = st.session_state.book_files[book_name]
        chapter_names = [c["name"] for c in book_data["chapters"]]

        selected = st.multiselect("Select chapters", chapter_names)
        num_cards = st.slider("Cards per chapter", min_value=3, max_value=15, value=5)

        if selected and st.button("Generate Flashcards"):
            chosen = [c for c in book_data["chapters"] if c["name"] in selected]
            cards = []
            with st.spinner("Generating flashcards with Claude..."):
                for chapter in chosen:
                    try:
                        new_cards = generate_flashcards(
                            book_data["pages"], chapter, num_cards, client
                        )
                        cards.extend(
                            {**card, "chapter": chapter["name"]} for card in new_cards
                        )
                    except Exception as e:
                        st.warning(f"Skipped '{chapter['name']}': {e}")
            st.session_state.flashcards = cards
            st.success(f"Generated {len(cards)} flashcards!")

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_chat, tab_fc = st.tabs(["Chat", "Flashcards"])

with tab_chat:
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
        sources = [
            f"{doc.metadata.get('source', 'unknown')} — {doc.page_content[:80]}..."
            for doc in chunks
        ]

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=(
                "You are a helpful assistant. Answer using only the context below.\n"
                "If the answer is not in the context, say 'I don't have that information.'\n\n"
                f"Context:\n{context}"
            ),
            messages=[{"role": "user", "content": question}]
        )

        answer = response.content[0].text
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("Sources"):
                for i, source in enumerate(sources):
                    st.caption(f"Chunk {i+1}: {source}")

with tab_fc:
    if not st.session_state.flashcards:
        st.info("Upload a book, select chapters in the sidebar, and click Generate Flashcards.")
    else:
        st.subheader(f"{len(st.session_state.flashcards)} Flashcards")
        if st.button("Clear Flashcards"):
            st.session_state.flashcards = []
            st.rerun()

        current_chapter = None
        for i, card in enumerate(st.session_state.flashcards):
            chapter_label = card.get("chapter", "")
            if chapter_label != current_chapter:
                st.markdown(f"#### {chapter_label}")
                current_chapter = chapter_label

            with st.expander(f"Card {i + 1} — {card['question'][:70]}..."):
                st.markdown(f"**Q:** {card['question']}")
                st.markdown(f"**A:** {card['answer']}")
