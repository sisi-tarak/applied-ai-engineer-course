# Project 01 — Company Knowledge Chatbot

> **What you build:** A production-ready RAG chatbot for any company's internal knowledge.
> **Time:** 6–8 hours | **Stack:** Python + LangChain + ChromaDB + FastAPI + Streamlit

**[← Back to Main](../../README.md)**

---

## What This Project Demonstrates

```
SKILLS YOU PROVE TO INTERVIEWERS:
  ✅ RAG pipeline (indexing + retrieval + generation)
  ✅ Multiple document format handling (PDF, DOCX, web)
  ✅ Production-quality code (error handling, logging, config)
  ✅ API development (FastAPI)
  ✅ UI development (Streamlit)
  ✅ Evaluation (RAGAS metrics)
  ✅ Deployment (Docker)

WHAT TO SAY IN INTERVIEWS:
  "I built a document chatbot that can ingest PDFs, Word docs,
   and websites, then answer questions with source citations.
   It handles 500+ documents with sub-2-second response time,
   uses ChromaDB for vector storage, and achieves 0.87 faithfulness
   score on our eval set. Deployed via Docker and FastAPI."
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    USER INTERFACE                         │
│              Streamlit Chat (app.py)                      │
└────────────────────────────┬─────────────────────────────┘
                             │ HTTP POST /chat
                             ▼
┌──────────────────────────────────────────────────────────┐
│                    FASTAPI SERVER                          │
│                     (api.py)                              │
│  Routes: /chat, /upload, /index, /health, /stats         │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                    RAG ENGINE                             │
│                  (rag_engine.py)                          │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │ Document │   │ Chunking │   │ Embedding│            │
│  │ Loader   │──▶│ Strategy │──▶│ + Index  │            │
│  └──────────┘   └──────────┘   └──────────┘            │
│                                      │                   │
│                                      ▼                   │
│                              ┌──────────────┐           │
│                              │  ChromaDB    │           │
│                              │ Vector Store │           │
│                              └──────┬───────┘           │
│                                     │ retrieve           │
│                                     ▼                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │  Answer  │◀──│   LLM    │◀──│ Retrieval│            │
│  │+ Sources │   │ Generate │   │  + Rank  │            │
│  └──────────┘   └──────────┘   └──────────┘            │
└──────────────────────────────────────────────────────────┘
```

---

## File Structure

```
project-01-company-chatbot/
├── README.md                    ← This file
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   ├── rag_engine.py            ← Core RAG logic
│   ├── document_processor.py    ← Multi-format document loading
│   ├── api.py                   ← FastAPI endpoints
│   └── config.py                ← Configuration management
│
├── app.py                       ← Streamlit chat interface
├── evaluate.py                  ← RAGAS evaluation script
│
├── sample_docs/                 ← Sample documents for testing
│   ├── sample_policy.pdf
│   └── sample_faq.txt
│
└── tests/
    ├── test_rag.py
    └── test_api.py
```

---

## Complete Implementation

### `src/config.py`

```python
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # LLM
    openai_api_key:     str = os.getenv("OPENAI_API_KEY", "")
    llm_model:          str = "gpt-4o-mini"
    llm_temperature:    float = 0.0

    # Embeddings
    embedding_model:    str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    chroma_db_path:     str = "./chroma_db"
    collection_name:    str = "company_knowledge"

    # Retrieval
    retrieval_k:        int = 4
    min_relevance_score: float = 0.45

    # Chunking
    chunk_size:         int = 500
    chunk_overlap:      int = 100

    # Company
    company_name:       str = os.getenv("COMPANY_NAME", "Company")
    fallback_message:   str = "I don't find relevant information in our documentation."

config = Config()
```

---

### `src/rag_engine.py`

```python
import os
import logging
from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Core RAG engine for company knowledge chatbot.
    Handles: indexing, retrieval, generation, conversation history.
    """

    def __init__(self):
        logger.info("Initializing RAG Engine...")

        self.embedder = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            api_key=config.openai_api_key
        )

        self.vectorstore = None
        self.conversation_history = []
        self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self):
        db_path = Path(config.chroma_db_path)
        if (db_path / "chroma.sqlite3").exists():
            self.vectorstore = Chroma(
                collection_name=config.collection_name,
                embedding_function=self.embedder,
                persist_directory=config.chroma_db_path
            )
            count = self.vectorstore._collection.count()
            logger.info(f"Loaded existing vectorstore with {count} chunks")
        else:
            logger.info("No existing vectorstore found. Will create on first index.")

    def index_documents(self, documents: list[Document]) -> dict:
        """Index a list of LangChain Documents into the vector store."""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedder,
                collection_name=config.collection_name,
                persist_directory=config.chroma_db_path
            )
        else:
            self.vectorstore.add_documents(chunks)

        return {
            "documents_indexed": len(documents),
            "chunks_created": len(chunks),
            "total_chunks": self.vectorstore._collection.count()
        }

    def _build_context(self, question: str) -> tuple[str, list]:
        """Retrieve and format context for LLM generation."""

        if self.vectorstore is None:
            return "", []

        results = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=config.retrieval_k
        )

        relevant = [
            (doc, score) for doc, score in results
            if score >= config.min_relevance_score
        ]

        if not relevant:
            return "", []

        context_parts = []
        sources = []
        for i, (doc, score) in enumerate(relevant, 1):
            source = doc.metadata.get("source", "Internal Document")
            page   = doc.metadata.get("page", "")
            page_str = f", Page {page}" if page else ""

            context_parts.append(
                f"[Reference {i}: {source}{page_str} | Relevance: {score:.2f}]\n"
                f"{doc.page_content}"
            )
            sources.append({
                "source":    source,
                "page":      page,
                "relevance": round(score, 3),
                "snippet":   doc.page_content[:150] + "..."
            })

        return "\n\n---\n\n".join(context_parts), sources

    def chat(self, question: str) -> dict:
        """Main chat function with conversation history support."""

        if self.vectorstore is None:
            return {
                "answer":  "No documents have been indexed yet. Please upload documents first.",
                "sources": [],
                "confidence": "none"
            }

        # Build context from vectorstore
        context, sources = self._build_context(question)

        if not context:
            self.conversation_history.append(
                {"role": "user",      "content": question}
            )
            self.conversation_history.append(
                {"role": "assistant", "content": config.fallback_message}
            )
            return {
                "answer":     config.fallback_message,
                "sources":    [],
                "confidence": "low"
            }

        # Build messages for LLM
        system_message = f"""You are a knowledgeable assistant for {config.company_name}.

RULES:
1. Answer ONLY using information from the provided context
2. Be specific — include numbers, dates, and percentages when available
3. Cite which reference your answer comes from (e.g., "Per Reference 1...")
4. If the question is not covered in the context, say exactly:
   "{config.fallback_message}"
5. Keep answers concise and professional

Context:
{context}"""

        messages = [{"role": "system", "content": system_message}]

        # Add conversation history (last 5 turns)
        for turn in self.conversation_history[-10:]:
            messages.append(turn)

        messages.append({"role": "user", "content": question})

        # Generate answer
        response = self.llm.invoke(messages)
        answer   = response.content

        # Update history
        self.conversation_history.append({"role": "user",      "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Determine confidence
        top_score  = sources[0]["relevance"] if sources else 0
        confidence = "high" if top_score > 0.7 else "medium" if top_score > 0.55 else "low"

        return {
            "answer":     answer,
            "sources":    sources,
            "confidence": confidence
        }

    def clear_history(self):
        self.conversation_history = []

    def get_stats(self) -> dict:
        count = self.vectorstore._collection.count() if self.vectorstore else 0
        return {
            "chunks_indexed":     count,
            "conversation_turns": len(self.conversation_history) // 2,
            "embedding_model":    config.embedding_model,
            "llm_model":          config.llm_model
        }
```

---

### `src/api.py`

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag_engine import RAGEngine
from src.document_processor import DocumentProcessor
import tempfile
import os

app = FastAPI(
    title=f"Company Knowledge API",
    description="RAG-powered chatbot for company knowledge base",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize engine once at startup
engine    = RAGEngine()
processor = DocumentProcessor()


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer:     str
    sources:    list
    confidence: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "stats": engine.get_stats()}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    if len(request.question) > 1000:
        raise HTTPException(400, "Question too long (max 1000 characters)")

    result = engine.chat(request.question)
    return ChatResponse(**result)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, DOCX, or TXT file and index it."""

    allowed_types = ["application/pdf", "text/plain",
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]

    if file.content_type not in allowed_types:
        raise HTTPException(400, f"File type not supported: {file.content_type}")

    # Save to temp file
    suffix = "." + file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        documents = processor.load(tmp_path)
        result    = engine.index_documents(documents)
        return {
            "message":  f"Successfully indexed {file.filename}",
            "filename": file.filename,
            **result
        }
    finally:
        os.unlink(tmp_path)


@app.delete("/history")
def clear_history():
    engine.clear_history()
    return {"message": "Conversation history cleared"}


@app.get("/stats")
def get_stats():
    return engine.get_stats()
```

---

### `app.py` — Streamlit Chat UI

```python
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Company Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Company Knowledge Assistant")
st.caption("Ask anything about our company policies, products, and procedures.")

# Sidebar — document upload
with st.sidebar:
    st.header("📄 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt"],
        help="Upload PDFs, Word docs, or text files"
    )

    if uploaded_file and st.button("Index Document"):
        with st.spinner("Indexing..."):
            response = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            )
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ Indexed! {data['chunks_created']} chunks created.")
            else:
                st.error(f"❌ Error: {response.text}")

    st.divider()

    # Stats
    try:
        stats = requests.get(f"{API_URL}/stats").json()
        st.metric("Documents Indexed", stats["chunks_indexed"])
        st.metric("Conversation Turns", stats["conversation_turns"])
    except:
        st.warning("API not connected")

    if st.button("🗑️ Clear History"):
        requests.delete(f"{API_URL}/history")
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander(f"📖 {len(message['sources'])} source(s)"):
                for src in message["sources"]:
                    st.write(f"• **{src['source']}** (relevance: {src['relevance']})")
                    st.caption(src["snippet"])

# Chat input
if question := st.chat_input("Ask a question..."):
    # Display user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/chat",
                json={"question": question}
            )

        if response.status_code == 200:
            data = response.json()

            # Color-code by confidence
            confidence_colors = {
                "high": "🟢", "medium": "🟡", "low": "🔴", "none": "⚫"
            }
            icon = confidence_colors.get(data["confidence"], "⚫")
            st.write(data["answer"])

            if data["sources"]:
                with st.expander(f"📖 {len(data['sources'])} source(s) | Confidence: {icon} {data['confidence']}"):
                    for src in data["sources"]:
                        st.write(f"• **{src['source']}** (relevance: {src['relevance']})")
                        st.caption(src["snippet"])

            st.session_state.messages.append({
                "role":    "assistant",
                "content": data["answer"],
                "sources": data["sources"]
            })
        else:
            st.error(f"Error: {response.text}")
```

---

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Start both API and Streamlit
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
```

---

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/sisi-tarak/applied-ai-engineer
cd applied-ai-engineer/projects/project-01-company-chatbot
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 3. Start the API server
uvicorn src.api:app --reload --port 8000

# 4. In a new terminal, start the UI
streamlit run app.py

# 5. Upload a document and start chatting!
# Open: http://localhost:8501

# OR run with Docker:
docker-compose up
```

---

## Evaluation

```bash
# Run RAGAS evaluation on your RAG system
python evaluate.py --test-file sample_docs/eval_questions.json
```

---

**[← Back to Main](../../README.md) | [Next: Project 2 →](../project-02-document-intelligence/README.md)**
