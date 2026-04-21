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

st.set_page_config(
    page_title="RAG-KnowledgePilot",
    page_icon="📖",
    layout="wide",
)

st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header ── */
.hero { padding: 1.5rem 0 0.5rem 0; }
.hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6C63FF, #48CAE4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero p { color: #888; font-size: 0.95rem; margin-top: 0.2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}
section[data-testid="stSidebar"] * { color: #cdd6f4 !important; }
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6C63FF, #48CAE4);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem;
    transition: opacity 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover { opacity: 0.85; }

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size: 0.95rem;
    font-weight: 600;
    color: #888 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #6C63FF !important;
    border-bottom: 2px solid #6C63FF !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 0.5rem;
    padding: 0.75rem 1rem;
}

/* ── Flashcard ── */
.fc-chapter {
    font-size: 1rem;
    font-weight: 700;
    color: #6C63FF;
    margin: 1.2rem 0 0.4rem 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.fc-card {
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.2s;
}
.fc-card:hover { border-color: #6C63FF; }
.fc-q { color: #cdd6f4; font-size: 0.95rem; margin-bottom: 0.5rem; }
.fc-a {
    color: #a6e3a1;
    font-size: 0.92rem;
    border-top: 1px solid #2a2a3e;
    padding-top: 0.5rem;
    margin-top: 0.5rem;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #555;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.5rem; }
.empty-state p { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

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

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "book_files" not in st.session_state:
    st.session_state.book_files = {}
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## RAG-KnowledgePilot")
    st.markdown("---")

    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if st.button("⚡ Ingest PDF"):
            with st.spinner("Processing..."):
                num_chunks, pages, is_book, chapters = ingest_uploaded_pdf(uploaded_file)
            st.success(f"✅ {num_chunks} chunks stored")
            if is_book:
                st.info(f"📚 Book detected — {len(chapters)} chapters")
                st.session_state.book_files[uploaded_file.name] = {
                    "pages": pages,
                    "chapters": chapters,
                }
            else:
                st.caption("📝 Regular document — flashcards not available")

    if st.session_state.book_files:
        st.markdown("---")
        st.markdown("### 🃏 Flashcard Generator")

        book_name = st.selectbox("Book", list(st.session_state.book_files.keys()), label_visibility="collapsed")
        book_data = st.session_state.book_files[book_name]
        chapter_names = [c["name"] for c in book_data["chapters"]]

        selected = st.multiselect("Select chapters", chapter_names, placeholder="Pick chapters...")
        num_cards = st.slider("Cards per chapter", min_value=3, max_value=15, value=5)

        if selected and st.button("✨ Generate Flashcards"):
            chosen = [c for c in book_data["chapters"] if c["name"] in selected]
            cards = []
            with st.spinner("Claude is generating flashcards..."):
                for chapter in chosen:
                    try:
                        new_cards = generate_flashcards(book_data["pages"], chapter, num_cards, client)
                        cards.extend({**card, "chapter": chapter["name"]} for card in new_cards)
                    except Exception as e:
                        st.warning(f"Skipped '{chapter['name']}': {e}")
            st.session_state.flashcards = cards
            st.success(f"🎉 {len(cards)} flashcards ready!")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>RAG-KnowledgePilot</h1>
  <p>Ask questions about your documents · Generate flashcards from books</p>
</div>
""", unsafe_allow_html=True)

# ── Chat input (must be outside tabs to work reliably) ────────────────────────
question = st.chat_input("Ask a question about your document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

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

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_chat, tab_fc = st.tabs(["💬  Chat", "🃏  Flashcards"])

with tab_chat:
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">💬</div>
          <p>Upload a PDF and start asking questions about it.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message:
                    with st.expander("📎 Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.caption(f"Chunk {i+1}: {source}")

with tab_fc:
    if not st.session_state.flashcards:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">🃏</div>
          <p>Upload a book PDF, select chapters in the sidebar,<br>and click <strong>Generate Flashcards</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### {len(st.session_state.flashcards)} Flashcards")
        with col2:
            if st.button("Clear", key="clear_fc"):
                st.session_state.flashcards = []
                st.rerun()

        current_chapter = None
        for i, card in enumerate(st.session_state.flashcards):
            chapter_label = card.get("chapter", "")
            if chapter_label != current_chapter:
                st.markdown(f'<div class="fc-chapter">{chapter_label}</div>', unsafe_allow_html=True)
                current_chapter = chapter_label

            st.markdown(f"""
            <div class="fc-card">
              <div class="fc-q">❓ {card['question']}</div>
              <div class="fc-a">✅ {card['answer']}</div>
            </div>
            """, unsafe_allow_html=True)
