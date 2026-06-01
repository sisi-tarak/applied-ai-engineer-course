# Module 04 — RAG Systems — The Complete Guide

> **Duration:** 10 hours | **Difficulty:** Intermediate | **Prerequisites:** Module 3

**[← Back to Main](../../README.md)**

---

## What Is RAG and Why Does It Dominate the Market?

```
PROBLEM WITH RAW LLMs:
  GPT-4 training cutoff: April 2023
  Your company's documents: updated daily
  GPT-4 knows: nothing about your company

  User: "What is our refund policy for premium accounts?"
  GPT-4: "I don't have access to your company policies."
  ← Useless.

RAG SOLUTION:
  User: "What is our refund policy for premium accounts?"
     ↓ embed query
  Vector search finds: policy_doc_page_3.pdf (similarity: 0.94)
     ↓ retrieve top 3 chunks
  LLM reads: [retrieved chunks] + question
  LLM answers: "Per your Premium Policy (Section 4.2):
                Premium accounts receive full refund within 30 days,
                50% refund from day 31–90, no refund after 90 days."
  ← Accurate. Current. Cited.

MARKET REALITY (2026):
  89% of Applied AI Engineer job postings require RAG experience
  RAG appears in 65% of all LLM production systems
  Average RAG project fee: ₹3L–15L
```

---

## RAG Architecture — The Complete Picture

```
INDEXING PIPELINE (runs once, or on document update):

Raw Documents
(PDFs, Word, Web, DBs, APIs)
         │
         ▼
   ┌─────────────┐
   │  Document   │  Parse PDFs, extract tables,
   │  Loading    │  handle images, clean text
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Chunking   │  Split into ~500 token pieces
   │  Strategy   │  with 50–100 token overlap
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Embedding  │  Convert each chunk to vector
   │  Model      │  text-embedding-3-small (OpenAI)
   └──────┬──────┘  all-MiniLM-L6-v2 (free, local)
          │
          ▼
   ┌─────────────┐
   │  Vector     │  Store embeddings + metadata
   │  Database   │  ChromaDB / Pinecone / Weaviate
   └─────────────┘

QUERY PIPELINE (runs on every user question):

User Question
         │
         ▼
   ┌─────────────┐
   │  Query      │  Same embedding model as indexing
   │  Embedding  │  (MUST be the same model)
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Similarity │  Find top-K most similar chunks
   │  Search     │  Cosine similarity, k=3 to 10
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Reranking  │  (Optional) Re-score retrieved chunks
   │  (optional) │  using cross-encoder for precision
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Prompt     │  Combine: system + context + question
   │  Assembly   │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  LLM        │  Generate answer grounded in context
   │  Generation │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Response   │  Answer + source citations
   │  + Sources  │
   └─────────────┘
```

---

## Implementation — Basic RAG in 50 Lines

```python
# File: 04_01_basic_rag.py
# Run: python 04_01_basic_rag.py
# Required: pip install langchain langchain-community chromadb
#           sentence-transformers langchain-openai

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI

load_dotenv()

# ── STEP 1: Your Documents ────────────────────────────────────
documents = [
    Document(
        page_content="""
        Refund Policy (Section 4.2 — Premium Accounts):
        Premium account holders may request full refunds within 30 days
        of purchase. Refunds requested between day 31 and day 90 receive
        50% of the purchase price. No refunds are issued after 90 days
        except in cases of verified technical failure on our part.
        Refunds are processed within 5–7 business days.
        """,
        metadata={"source": "premium_policy.pdf", "section": "4.2", "page": 12}
    ),
    Document(
        page_content="""
        Shipping Policy (Section 2.1):
        Standard shipping: 5–7 business days, free on orders above ₹999.
        Express shipping: 2–3 business days, ₹149 flat.
        Same-day delivery: Available in Mumbai, Delhi, Bengaluru only.
        Orders placed before 2 PM are shipped same day.
        International shipping: Available to 50+ countries, 7–14 days.
        """,
        metadata={"source": "shipping_policy.pdf", "section": "2.1", "page": 5}
    ),
    Document(
        page_content="""
        Customer Support Escalation Matrix:
        Level 1: Chatbot handles — billing queries, order status, basic FAQs.
        Level 2: Support agent — complaints, refund requests, account issues.
        Level 3: Senior agent — fraud, legal threats, VIP customers.
        Level 4: Manager — unresolved complaints after 48 hours.
        Response SLA: Level 1 = instant, Level 2 = 4 hrs, Level 3 = 1 hr.
        """,
        metadata={"source": "support_matrix.pdf", "section": "3.0", "page": 8}
    ),
]

# ── STEP 2: Split into Chunks ─────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")

# ── STEP 3: Create Embeddings + Vector Store ──────────────────
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# Downloads ~90MB model first time, then cached locally

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedder,
    persist_directory="./company_knowledge_db"
)
print("✅ Knowledge base created and saved!")

# ── STEP 4: RAG Query Function ────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def ask(question: str, k: int = 3) -> dict:
    """
    Ask a question against your company knowledge base.
    Returns answer + sources.
    """
    # Retrieve relevant chunks
    relevant_docs = vectorstore.similarity_search(question, k=k)

    # Build context
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # Generate answer
    prompt = f"""You are a helpful customer support assistant.
Answer the question using ONLY the information provided below.
If the answer is not in the context, say: "I don't find that in our documentation."
Always cite which source your answer came from.

Context:
{context}

Question: {question}

Answer:"""

    answer = llm.invoke(prompt).content

    return {
        "answer": answer,
        "sources": [doc.metadata.get("source") for doc in relevant_docs],
        "chunks_retrieved": len(relevant_docs)
    }

# ── STEP 5: Test It ───────────────────────────────────────────
test_questions = [
    "Can I get a refund after 2 months?",
    "How long does express shipping take?",
    "Who handles fraud cases?",
    "What is your WiFi policy?",  # Not in our docs — should say unknown
]

print("\n" + "="*60)
for question in test_questions:
    result = ask(question)
    print(f"\n❓ Question: {question}")
    print(f"💬 Answer: {result['answer']}")
    print(f"📖 Sources: {result['sources']}")
    print("-" * 60)
```

---

## Implementation — Production-Grade RAG

```python
# File: 04_02_production_rag.py
# This is what you deliver to clients — not the basic version

import os
import hashlib
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WebBaseLoader
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()


class ProductionRAG:
    """
    Production-grade RAG system with:
    - Multiple document format support (PDF, DOCX, TXT, Web)
    - Smart chunking with metadata preservation
    - Source citation in answers
    - Conversation history support
    - Query reformulation for better retrieval
    - Fallback handling
    """

    def __init__(self, db_path: str = "./rag_db", company_name: str = "Company"):
        self.db_path    = db_path
        self.company    = company_name
        self.embedder   = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm        = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore = None
        self.history     = []

        # Load existing DB if it exists
        if Path(f"{db_path}/chroma.sqlite3").exists():
            self.vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=self.embedder
            )
            print(f"✅ Loaded existing knowledge base from {db_path}")

    def load_documents(self, source: str) -> list:
        """Load from file path or URL."""
        if source.startswith("http"):
            loader = WebBaseLoader(source)
        elif source.endswith(".pdf"):
            loader = PyPDFLoader(source)
        elif source.endswith(".docx"):
            loader = Docx2txtLoader(source)
        else:
            loader = TextLoader(source)

        docs = loader.load()
        print(f"  Loaded {len(docs)} pages from {source}")
        return docs

    def index_documents(self, sources: list, force_reindex: bool = False):
        """
        Index documents into vector store.
        Tracks which files are indexed to avoid duplicates.
        """
        index_log = Path(f"{self.db_path}/indexed_files.json")
        indexed   = json.loads(index_log.read_text()) if index_log.exists() else {}

        splitter  = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            add_start_index=True
        )

        all_chunks = []
        for source in sources:
            # Check if already indexed (using file hash)
            file_hash = hashlib.md5(source.encode()).hexdigest()
            if file_hash in indexed and not force_reindex:
                print(f"  ⏭️ Already indexed: {source}")
                continue

            print(f"  📄 Indexing: {source}")
            docs   = self.load_documents(source)
            chunks = splitter.split_documents(docs)

            # Add source metadata to each chunk
            for chunk in chunks:
                chunk.metadata["file_source"] = source
                chunk.metadata["chunk_hash"]  = hashlib.md5(
                    chunk.page_content.encode()
                ).hexdigest()

            all_chunks.extend(chunks)
            indexed[file_hash] = {"source": source, "chunks": len(chunks)}

        if not all_chunks:
            print("No new documents to index.")
            return

        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embedder,
                persist_directory=self.db_path
            )
        else:
            self.vectorstore.add_documents(all_chunks)

        # Save index log
        index_log.parent.mkdir(exist_ok=True)
        index_log.write_text(json.dumps(indexed, indent=2))
        print(f"✅ Indexed {len(all_chunks)} new chunks. Total: {self.vectorstore._collection.count()}")

    def _reformulate_query(self, query: str, history: list) -> str:
        """
        Reformulate ambiguous queries using conversation history.
        Example: "What about the refund?" → "What is the refund policy?"
        """
        if not history:
            return query

        history_text = "\n".join([
            f"User: {h['user']}\nAssistant: {h['assistant'][:100]}..."
            for h in history[-3:]
        ])

        prompt = f"""Given this conversation history and a follow-up question,
rewrite the question to be self-contained and clear.
Return ONLY the rewritten question, nothing else.

History:
{history_text}

Follow-up question: {query}

Rewritten question:"""

        return self.llm.invoke(prompt).content.strip()

    def ask(self, question: str, k: int = 4,
            use_history: bool = True) -> dict:
        """Main query function with full features."""

        if self.vectorstore is None:
            return {"error": "No documents indexed yet. Call index_documents() first."}

        # Step 1: Reformulate query if needed
        effective_query = question
        if use_history and self.history:
            effective_query = self._reformulate_query(question, self.history)
            if effective_query != question:
                print(f"  🔄 Reformulated: '{effective_query}'")

        # Step 2: Retrieve relevant chunks
        results = self.vectorstore.similarity_search_with_relevance_scores(
            effective_query, k=k
        )

        # Filter by relevance threshold
        relevant = [(doc, score) for doc, score in results if score > 0.4]

        if not relevant:
            return {
                "answer": "I couldn't find relevant information in the documentation for this question.",
                "sources": [],
                "confidence": "low"
            }

        # Step 3: Build context with source tracking
        context_parts = []
        sources_used  = []
        for i, (doc, score) in enumerate(relevant, 1):
            source = doc.metadata.get("file_source", "Unknown")
            page   = doc.metadata.get("page", "N/A")
            context_parts.append(
                f"[Source {i}: {source}, Page {page}, Relevance: {score:.2f}]\n"
                f"{doc.page_content}"
            )
            sources_used.append({"file": source, "page": page, "score": round(score, 3)})

        context = "\n\n---\n\n".join(context_parts)

        # Step 4: Generate grounded answer
        prompt = f"""You are a knowledgeable assistant for {self.company}.
Your answers must be:
1. Grounded ONLY in the provided context
2. Accurate and specific (include numbers, dates, percentages when available)
3. Clear and concise
4. End with the source citation

If the context doesn't contain the answer, say:
"I don't find information about this in our current documentation."

Context:
{context}

Question: {question}

Answer (cite sources):"""

        answer = self.llm.invoke(prompt).content

        # Step 5: Update history
        self.history.append({"user": question, "assistant": answer})
        if len(self.history) > 10:
            self.history = self.history[-10:]  # keep last 10 turns

        return {
            "answer":     answer,
            "sources":    sources_used,
            "confidence": "high" if relevant[0][1] > 0.7 else "medium"
        }


# ── USAGE EXAMPLE ─────────────────────────────────────────────
if __name__ == "__main__":
    # Initialize
    rag = ProductionRAG(db_path="./company_rag", company_name="Webortex")

    # Index documents (PDF, DOCX, web pages, text files)
    # rag.index_documents([
    #     "company_policy.pdf",
    #     "product_catalog.docx",
    #     "https://company.com/faq",
    #     "support_docs.txt"
    # ])

    # For demo: create a simple text file and index it
    with open("demo_policy.txt", "w") as f:
        f.write("""
        WEBORTEX AI SERVICES POLICY

        Service Packages:
        - Starter: ₹25,000/month — Includes chatbot setup, 10K queries/month
        - Professional: ₹75,000/month — Full RAG system, 50K queries/month
        - Enterprise: Custom pricing — Fine-tuned models, unlimited queries

        SLA Commitments:
        - 99.9% uptime guarantee for Professional and Enterprise
        - Support response: 4 hours for Professional, 1 hour for Enterprise
        - Monthly performance reports included

        Data Privacy:
        - All data processed within India (Mumbai data center)
        - Zero data sharing with third parties
        - Data deleted within 30 days of contract termination
        """)

    rag.index_documents(["demo_policy.txt"])

    # Multi-turn conversation test
    print("\n💬 Starting conversation:\n")

    questions = [
        "What are your service packages?",
        "What's the uptime guarantee?",
        "What about their pricing?",    # ambiguous — uses history
    ]

    for q in questions:
        result = rag.ask(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print()
```

---

## Chunking Strategies — The Most Important RAG Decision

```python
# File: 04_03_chunking_strategies.py

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    SentenceTransformersTokenTextSplitter
)

sample_text = """
LOAN POLICY — PERSONAL LOANS

Section 1: Eligibility Criteria
Applicants must be between 21 and 65 years of age.
Minimum monthly income: ₹25,000 for salaried, ₹35,000 for self-employed.
CIBIL score requirement: minimum 700.
Employment stability: minimum 2 years at current employer.

Section 2: Loan Amount and Tenure
Minimum loan amount: ₹50,000
Maximum loan amount: ₹25,00,000 (subject to income and credit score)
Tenure options: 12, 24, 36, 48, or 60 months
Maximum EMI to income ratio: 50%

Section 3: Interest Rates
Standard rate: 10.5% to 18% per annum (based on credit profile)
Processing fee: 2% of loan amount (minimum ₹2,000)
Prepayment charges: Nil after 12 months, 2% before 12 months
"""

# Strategy 1: Fixed-size chunking (fast, simple, baseline)
fixed_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
fixed_chunks = fixed_splitter.split_text(sample_text)
print(f"Fixed chunking: {len(fixed_chunks)} chunks")

# Strategy 2: Markdown-aware chunking (best for structured docs)
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"), ("##", "h2"), ("###", "h3")
    ]
)
# Convert to markdown-style headers first
md_text = sample_text.replace("Section 1:", "## Section 1:").replace("Section 2:", "## Section 2:").replace("Section 3:", "## Section 3:")
md_chunks = md_splitter.split_text(md_text)
print(f"Markdown chunking: {len(md_chunks)} chunks (preserves structure)")

# Strategy 3: Semantic chunking (best quality, slowest)
# Splits at natural semantic boundaries, not arbitrary character counts
# Requires: pip install langchain-experimental
# from langchain_experimental.text_splitter import SemanticChunker
# semantic_chunks = SemanticChunker(embedder).split_text(sample_text)

# ── CHUNKING DECISION GUIDE ───────────────────────────────────
print("""
WHEN TO USE WHICH CHUNKING:

Fixed-size (RecursiveCharacterTextSplitter):
  ✅ General purpose, works for most cases
  ✅ Fast, predictable
  ❌ Can cut sentences mid-thought

Markdown-header (MarkdownHeaderTextSplitter):
  ✅ Technical docs, policies, manuals with headers
  ✅ Preserves document structure as metadata
  ❌ Only works when document has consistent headers

Semantic (SemanticChunker):
  ✅ Best retrieval quality
  ✅ Splits at natural meaning boundaries
  ❌ Slow (requires embeddings during chunking)
  ❌ Chunk sizes can be unpredictable

Recommended sizes:
  chunk_size=500, chunk_overlap=100  → General knowledge base
  chunk_size=300, chunk_overlap=50   → FAQ, short policy docs
  chunk_size=800, chunk_overlap=150  → Long research papers
""")
```

---

## RAG Evaluation with RAGAS

```python
# File: 04_04_rag_evaluation.py
# Required: pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset

# Test dataset: questions, ground truth, and what your RAG returned
test_data = {
    "question": [
        "What is the maximum loan amount?",
        "What CIBIL score is required?",
        "Can I prepay before 12 months?",
    ],
    "answer": [   # What your RAG system answered
        "The maximum loan amount is ₹25,00,000 subject to income and credit profile.",
        "A minimum CIBIL score of 700 is required.",
        "Yes, but prepayment before 12 months incurs a 2% charge.",
    ],
    "contexts": [   # Chunks your RAG retrieved (list of lists)
        ["Maximum loan amount: ₹25,00,000 (subject to income and credit score)"],
        ["CIBIL score requirement: minimum 700."],
        ["Prepayment charges: Nil after 12 months, 2% before 12 months"],
    ],
    "ground_truth": [   # Correct answers (for measuring recall)
        "Maximum loan amount is ₹25,00,000.",
        "Minimum CIBIL score required is 700.",
        "Prepayment before 12 months has a 2% charge; free after 12 months.",
    ]
}

dataset = Dataset.from_dict(test_data)

# Evaluate
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,       # Is the answer supported by the retrieved context?
        answer_relevancy,   # Is the answer relevant to the question?
        context_recall,     # Did we retrieve the right chunks?
        context_precision,  # Are the retrieved chunks actually useful?
    ]
)

print("RAG Evaluation Results:")
print(f"  Faithfulness:       {results['faithfulness']:.3f}")
print(f"  Answer Relevancy:   {results['answer_relevancy']:.3f}")
print(f"  Context Recall:     {results['context_recall']:.3f}")
print(f"  Context Precision:  {results['context_precision']:.3f}")
print()
print("Interpretation:")
print("  > 0.9  → Production ready")
print("  0.7–0.9 → Needs improvement (check chunking & retrieval)")
print("  < 0.7  → Major issues (bad embedding or chunking)")
```

---

## Common RAG Failures and Fixes

```
FAILURE 1: Wrong chunks retrieved (low context precision)
─────────────────────────────────────────────────────────
Symptom: Answer uses irrelevant information
Fix 1: Improve chunking — increase overlap, use semantic chunking
Fix 2: Add reranking (CrossEncoderReranker) after initial retrieval
Fix 3: Use hybrid search (dense + sparse/BM25)

FAILURE 2: Answer contradicts source (low faithfulness)
────────────────────────────────────────────────────────
Symptom: LLM makes up information not in retrieved chunks
Fix 1: Lower temperature to 0
Fix 2: Add explicit instruction: "Answer ONLY from the context"
Fix 3: Add self-verification step after generation
Fix 4: Use smaller, more focused chunks

FAILURE 3: Misses information that exists (low context recall)
───────────────────────────────────────────────────────────────
Symptom: "I don't find that" but document has the answer
Fix 1: Increase k (retrieve more chunks)
Fix 2: Use query expansion — generate multiple variants of query
Fix 3: Check embedding model quality on your domain

FAILURE 4: Slow retrieval (latency issues)
───────────────────────────────────────────
Symptom: > 2 seconds for retrieval
Fix 1: Use Pinecone instead of ChromaDB (for production scale)
Fix 2: Cache embeddings for common queries
Fix 3: Use approximate nearest neighbor search (FAISS)
Fix 4: Reduce embedding dimensions (use MRL models)
```

---

## Resources

| Resource | Type | Time | What you get |
|----------|------|------|--------------|
| [RAG from Scratch — LangChain](https://youtube.com/watch?v=sVcwVQRHIc8) | 📺 YouTube | 2 hrs | Build complete RAG |
| [Advanced RAG Techniques](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6) | 📖 Blog | 1 hr | 12 RAG improvements |
| [RAGAS Documentation](https://docs.ragas.io) | 📖 Docs | 30 min | RAG evaluation |
| [ChromaDB Docs](https://docs.trychroma.com) | 📖 Docs | 30 min | Local vector DB |
| [Pinecone Learning Center](https://www.pinecone.io/learn/) | 📖 Docs | 2 hrs | Production vector DB |
| [RAG vs Fine-tuning — IBM](https://www.ibm.com/think/topics/rag-vs-fine-tuning-vs-prompt-engineering) | 📖 Blog | 20 min | When to use each |
| [LlamaIndex Docs](https://docs.llamaindex.ai) | 📖 Docs | 2 hrs | Alternative RAG framework |

---

**[← Module 3](../03-prompt-engineering/README.md) | [Back to Main](../../README.md) | [Next: Module 5 — Fine-tuning →](../05-fine-tuning/README.md)**
