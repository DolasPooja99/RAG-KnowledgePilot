# RAG-KnowledgePilot

Ask questions about your documents using AI. Upload any PDF and get accurate, 
source-cited answers powered by RAG (Retrieval-Augmented Generation) and Claude.

## What I built and why

I built this project to understand RAG from the ground up — not just use a library, 
but understand every step: how text becomes embeddings, how semantic search works, 
and how to ground an LLM's answers in real documents instead of its training data.

## Demo

> "What is Pooja's work experience?"
> → Claude retrieves the 3 most relevant chunks from the uploaded resume and answers 
> using only that context, with source citations showing exactly which part of the 
> document it used.

## Architecture

```
PDF Upload → Chunk (500 tokens, 50 overlap) → Embed (all-MiniLM-L6-v2)
                                                        ↓
                                                   pgvector (PostgreSQL)
                                                        ↓
Question → Embed → Semantic Search → Top 9 Chunks → Claude (Haiku) → Answer + Sources
```

## Tech stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| LangChain | RAG pipeline orchestration |
| HuggingFace all-MiniLM-L6-v2 | Local embedding model — no API cost |
| pgvector | Vector storage and semantic search in PostgreSQL |
| Anthropic Claude API | Answer generation grounded in retrieved context |
| Streamlit | Chat UI with sidebar upload and source citations |

## Features

- Upload multiple PDFs — each tracked by filename
- Duplicate detection — same file won't be ingested twice
- Source citations — every answer shows which chunks were used and from which file
- Multi-document Q&A — ask questions across all uploaded documents at once
- Clear chat — reset conversation without losing ingested documents

## What I learned

- How embeddings represent meaning as vectors and why similar meaning = similar numbers
- Chunking strategy matters more than model choice — chunk size and overlap directly affect answer quality
- The difference between keyword search and semantic search
- How to ground LLM answers in real data to prevent hallucinations
- How pgvector turns PostgreSQL into a vector database with one extension

## Setup

```bash
git clone https://github.com/DolasPooja99/knowledgepilot.git
cd knowledgepilot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your-key-here
```

Set up PostgreSQL:
```bash
createdb knowledgepilot
psql knowledgepilot -c "CREATE EXTENSION vector;"
```

Run:
```bash
streamlit run src/app.py
```

## Project structure

```
knowledgepilot/
├── src/
│   ├── app.py          ← Streamlit UI + upload handling
│   ├── ingest.py       ← PDF → chunks → embeddings → pgvector
│   ├── retriever.py    ← semantic search against vector store
│   └── agent.py        ← Claude integration with retrieved context
├── docs/               ← sample PDFs for testing
├── .env                ← API keys (not committed)
└── requirements.txt


