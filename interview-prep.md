# Applied AI Engineer — Interview Prep Guide

> **50 real interview questions with answers**
> Based on actual interviews at Sarvam AI, Krutrim, Freshworks, and funded Indian AI startups (2025–2026)

**[← Back to Main](../README.md)**

<br/>

## How Interviews Are Structured

```
TYPICAL APPLIED AI ENGINEER INTERVIEW:

Round 1 — Technical Screening (45 min, phone/video)
  → 5-10 conceptual questions (RAG, fine-tuning, agents)
  → 1 take-home or live coding task

Round 2 — Deep Technical (60–90 min)
  → System design: "Design a RAG system for a hospital"
  → Code review or live coding
  → Debug a broken agent

Round 3 — Portfolio Review (45 min)
  → Walk through 2-3 projects you've built
  → Architecture decisions you made
  → What you'd do differently

Round 4 — Culture + Leadership (30–45 min)
  → How you work with non-technical stakeholders
  → How you handle ambiguity
  → Career goals
```

<br/>

## Section 1 — RAG Questions (Most Asked)

### Q1: Explain RAG to a non-technical person.

**Answer:**
Imagine hiring a smart employee who knows everything in general but doesn't know your company's specific policies. RAG is like giving them a binder with all your company documents to refer to while answering questions. The AI searches the binder, finds the relevant page, reads it, and gives a specific answer — instead of guessing.

<br/>

### Q2: What is the difference between dense and sparse retrieval?

**Answer:**
```
Dense retrieval (semantic):
  - Convert text to vectors using embedding model
  - Find similar vectors using cosine similarity
  - Understands meaning: "car" matches "automobile"
  - Requires embedding model

Sparse retrieval (keyword / BM25):
  - Match exact keywords (like traditional search)
  - Fast, no embedding needed
  - Fails on synonyms and paraphrasing
  - Works well when users know exact terms

Hybrid search (best of both):
  - Run both, combine results
  - Default recommendation for production RAG
  - Pinecone, Weaviate, OpenSearch all support this
```

<br/>

### Q3: What happens when your RAG system retrieves wrong chunks?

**Answer:**
This is a precision problem. Solutions in order of priority:
1. **Add reranking** — after initial retrieval, use a cross-encoder (like `cross-encoder/ms-marco-MiniLM-L6-v2`) to re-score and filter
2. **Improve chunking** — chunks that mix topics confuse retrieval; split at semantic boundaries
3. **Add metadata filtering** — filter by document type, date, or section before semantic search
4. **Query expansion** — generate 3 variants of the query, retrieve for all, deduplicate

<br/>

### Q4: How do you handle the "I don't know" problem in RAG?

**Answer:**
The LLM should say "not in documentation" instead of hallucinating. Three approaches:
1. **Relevance threshold**: Only pass chunks to LLM if similarity score > 0.7
2. **Explicit instruction in system prompt**: "If the answer is not in the context, say exactly: I don't find that in our documentation"
3. **Faithfulness evaluation**: Run RAGAS faithfulness metric in production; alert when score drops

<br/>

### Q5: Design a RAG system for a 10,000-employee company with 500,000 documents.

**Answer (system design format):**
```
REQUIREMENTS CLARIFICATION:
  - Document types: PDFs, Word, Excel, emails, Slack
  - Languages: English + 3 regional languages?
  - Query volume: how many per day?
  - Latency requirement: < 2 seconds?
  - Access control: department-level permissions?

ARCHITECTURE:
  Ingestion:
    → Apache Tika for parsing all formats
    → Language detection per chunk
    → Department metadata tagged at indexing
    → Scheduled re-indexing for updated docs (weekly)

  Storage:
    → Pinecone with namespaces (one per department)
    → PostgreSQL for document metadata and access control
    → Redis for caching frequent queries

  Retrieval:
    → Access control filter applied BEFORE semantic search
    → Hybrid: BM25 + dense embeddings
    → Cross-encoder reranking (top 20 → top 5)

  Generation:
    → Claude Sonnet for cost-quality balance
    → Streaming response for UX
    → Source citation mandatory

  Evaluation:
    → RAGAS in staging environment
    → Human review queue for low-confidence answers
    → A/B testing for retrieval improvements
```

<br/>

## Section 2 — Fine-tuning Questions

### Q6: When would you choose fine-tuning over RAG?

**Answer:**
I use a decision framework:
- **Factual recall needed?** → RAG (fine-tuning can hallucinate facts)
- **Style/tone consistency needed?** → Fine-tuning
- **Data changes frequently?** → RAG (no retraining needed)
- **Specific format required every time?** → Fine-tuning (e.g., SOAP notes, legal citations)
- **< 500 examples available?** → Prompting or RAG; fine-tuning won't generalize well

Often the answer is **both**: RAG for current information + fine-tuned model for the right style and format.

<br/>

### Q7: What is LoRA and why is it preferred?

**Answer:**
LoRA (Low-Rank Adaptation) fine-tunes a model by adding small trainable matrices (A and B) to the original frozen weight matrices (W). Instead of updating 7 billion parameters, you only update ~20 million LoRA parameters.

Why preferred:
- **Memory**: 16GB GPU vs 80GB+ for full fine-tuning
- **Cost**: $1–5 vs $500–5000 for full fine-tuning
- **Speed**: Hours vs weeks
- **Quality**: Within 1–3% of full fine-tuning on most tasks
- **Reversibility**: Remove LoRA weights to restore original model

QLoRA adds 4-bit quantization of the base model — same concept, even more memory efficient.

<br/>

### Q8: How many training examples do you need for fine-tuning?

**Answer:**
```
Rule of thumb:
  Simple formatting/style:  50–200 examples
  Domain adaptation:        500–2,000 examples
  New capability:           2,000–10,000 examples
  General instruction:      10,000+ examples

Quality > Quantity:
  I'd rather have 200 expert-reviewed examples
  than 2,000 auto-generated mediocre ones.

Always:
  - Manually review 10% of your dataset
  - Create separate validation set (10%)
  - Evaluate on held-out test set before deployment
```

<br/>

### Q9: How do you evaluate a fine-tuned model?

**Answer:**
Three levels:
1. **Automatic metrics**: BLEU, ROUGE for generation tasks; accuracy/F1 for classification. Fast, objective.
2. **LLM-as-judge**: Use GPT-4 to score outputs 1–5 on criteria like accuracy, relevance, format. More aligned with human judgment.
3. **Human evaluation**: Have domain experts (doctors, lawyers) rate 100 outputs. Most accurate, most expensive. Required before production.

I build an eval set from real user queries + expert-verified answers before training starts.

<br/>

## Section 3 — Agent Questions

### Q10: Explain the ReAct pattern.

**Answer:**
ReAct = Reason + Act. The agent alternates between:
- **Thought**: Reasoning about what to do next
- **Action**: Calling a tool or API
- **Observation**: Reading the result

```
User: "What is Apple's current stock price in rupees?"

THOUGHT: I need Apple's current USD price and current USD/INR rate
ACTION: search_web("Apple AAPL stock price today")
OBSERVATION: AAPL = $187.30

THOUGHT: Now I need the exchange rate
ACTION: get_exchange_rate("USD", "INR")
OBSERVATION: 1 USD = 83.42 INR

THOUGHT: I can calculate now: 187.30 × 83.42 = 15,624.97
FINAL ANSWER: Apple stock is currently ₹15,624.97
```

This is better than a single LLM call because the agent verifies real-time data instead of guessing.

<br/>

### Q11: How do you prevent an agent from running infinitely?

**Answer:**
Three safeguards:
1. **max_iterations**: Hard limit on tool call loop (e.g., 10 iterations)
2. **timeout**: Kill agent if running > 30 seconds
3. **budget cap**: Track API calls and cost; stop if limit exceeded
4. **human-in-the-loop**: For destructive actions (deleting files, sending emails), require explicit user confirmation before executing
5. **action whitelist**: Define exactly which tools the agent can call — no arbitrary code execution

<br/>

### Q12: What is the difference between LangGraph and CrewAI?

**Answer:**
```
CrewAI:
  - Role-based agents (Researcher, Writer, Reviewer)
  - Good for document creation workflows
  - Sequential or hierarchical process
  - Simple to set up
  - Less control over exact execution flow

LangGraph:
  - State machine model
  - Conditional branching (if/else between steps)
  - Loops and retries
  - Human-in-the-loop checkpoints
  - More complex but more controllable
  - Better for production systems

Use CrewAI when: Team of agents creates a document/report
Use LangGraph when: Complex workflow with conditions, retries, human approval
```

<br/>

## Section 4 — System Design Questions

### Q13: Design an AI system for a 100-person customer support team

**Answer:**
```
COMPONENTS:

1. Smart Triage Agent
   → Classify ticket: billing/technical/general
   → Assess urgency: P1/P2/P3
   → Route to right team/agent

2. Agent Assist
   → RAG over past resolved tickets + knowledge base
   → Suggest response to human agent (not auto-send)
   → Human reviews and sends

3. Auto-resolve for simple tickets
   → Order status, tracking, basic FAQs
   → Only if confidence > 0.95
   → Human oversight on 5% sample

4. Quality Monitor
   → CSAT prediction before survey
   → Flag low-quality responses
   → Identify knowledge gaps

METRICS:
   First response time: target < 4 hours (vs 8 hours current)
   Resolution rate: target 60% auto-resolved
   CSAT target: > 4.2/5
   Cost per ticket: target 40% reduction
```

<br/>

## Section 5 — Code Questions (Live Coding)

### Q14: Write a function to chunk a document for RAG

```python
def chunk_document(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Split text into overlapping chunks for RAG indexing.

    Args:
        text: Input document text
        chunk_size: Target chunk size in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or len(text) == 0:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If not at the end, find a good break point
        if end < len(text):
            # Try to break at a sentence boundary
            for boundary in [". ", ".\n", "\n\n", "\n", " "]:
                break_at = text.rfind(boundary, start, end)
                if break_at > start + chunk_size // 2:  # Found a good break
                    end = break_at + len(boundary)
                    break

        chunk = text[start:end].strip()
        if chunk:  # Don't add empty chunks
            chunks.append(chunk)

        start = end - overlap  # Back up by overlap amount

    return chunks


# Test
sample = "Sentence one. Sentence two. Sentence three. Sentence four."
chunks = chunk_document(sample, chunk_size=30, overlap=10)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: '{chunk}'")
```

<br/>

### Q15: Debug this RAG system — it always returns wrong answers

```python
# BROKEN CODE — find and fix the bugs

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# BUG 1: Using different embedding models for indexing and querying
indexing_embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
query_embedder    = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")  # ← WRONG

vectorstore = Chroma(embedding_function=indexing_embedder)

def broken_query(question: str) -> str:
    # BUG 2: Using wrong embedder for query
    results = vectorstore.similarity_search(
        question,
        k=1,                          # BUG 3: Only retrieving 1 chunk (too few)
        embedding=query_embedder       # BUG 4: Mismatched embedder
    )

    # BUG 5: No relevance threshold — passes irrelevant chunks to LLM
    context = "\n".join([doc.page_content for doc in results])

    # BUG 6: Temperature too high for factual Q&A
    llm = ChatOpenAI(temperature=1.0)   # ← Should be 0

    return llm.invoke(f"Answer: {question}\nContext: {context}").content
    # BUG 7: Instruction too vague — model will ignore context and guess


# FIXED VERSION:
def fixed_query(question: str, vectorstore, llm) -> str:
    # Fix 1: Same embedder as indexing (handled by vectorstore)
    # Fix 2: Retrieve more chunks
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(question, k=4)

    # Fix 3: Filter by relevance threshold
    relevant = [(doc, score) for doc, score in results_with_scores if score > 0.5]

    if not relevant:
        return "I don't find relevant information in the documentation."

    # Fix 4: Clear context formatting
    context = "\n\n".join([doc.page_content for doc, _ in relevant])

    # Fix 5: Clear instruction (temperature=0 set on llm creation)
    prompt = f"""Answer using ONLY the context below. 
If not in context, say "Not found in documentation."

Context: {context}

Question: {question}
Answer:"""

    return llm.invoke(prompt).content
```

<br/>

## Section 6 — Behavioral Questions

### Q16: Tell me about a time a model gave wrong outputs in production. How did you handle it?

**Framework for answering:**
```
SITUATION: What AI system and what went wrong
TASK:      What was your responsibility
ACTION:    
  1. How you detected it (monitoring, user report)
  2. How you diagnosed root cause
  3. How you fixed it (hot fix vs proper fix)
  4. How you prevented recurrence
RESULT:    Impact — reduction in error rate, recovered users
```

<br/>

### Q17: How do you explain AI limitations to a non-technical client?

**Sample answer:**
"I tell clients that AI is like a very well-read expert who sometimes confabulates — like a doctor who confidently gives a wrong diagnosis. So we always build human review for high-stakes decisions. We also set up monitoring so if the AI starts giving unusual answers, we catch it before users do. The goal isn't to replace human judgment — it's to handle the repetitive 80% so humans can focus on the nuanced 20%."

<br/>

## Quick Reference — Common Interview Topics

```
TOPIC                    DEPTH EXPECTED          MODULE
─────────────────────────────────────────────────────────
RAG architecture         Draw + explain          Module 4
Chunking strategies      Name 3 + when to use    Module 4
Vector databases         ChromaDB vs Pinecone    Module 4
Embedding models         Free vs OpenAI trade-offs Module 4
RAGAS metrics            Explain 4 metrics       Module 7
LoRA vs QLoRA            Memory comparison        Module 5
When to fine-tune        Decision framework       Module 5
ReAct pattern            Draw the loop           Module 6
LangGraph vs CrewAI      Use cases for each      Module 6
FastAPI deployment        Build a basic endpoint  Module 8
Docker containerization  Dockerfile basics       Module 8
```

<br/>

## 30-Day Interview Prep Plan

```
WEEK 1 — Foundation
  Day 1-2:  Revise Module 1-4 (RAG core)
  Day 3-5:  Build Project 1 from scratch, no help
  Day 6:    Record yourself explaining RAG in 5 min
  Day 7:    Practice Q1-Q15 answers out loud

WEEK 2 — Advanced
  Day 8-10:  Revise Module 5 (fine-tuning)
  Day 11-12: Build a fine-tuning run (even small scale)
  Day 13-14: Practice design questions (Q5, Q13)

WEEK 3 — Projects + Portfolio
  Day 15-18: Polish GitHub README for all 4 projects
  Day 19-21: Record 2-minute demo videos per project
  Day 22:    Update LinkedIn with projects
  Day 23-24: Practice explaining projects in 5 minutes

WEEK 4 — Mock Interviews
  Day 25-26: Mock technical interview (ask a friend)
  Day 27:    Record yourself answering Q10-Q15
  Day 28:    Apply to 5 companies
  Day 29-30: Follow up, prep for specific companies
```

<br/>

**[← Back to Main](../README.md)**
