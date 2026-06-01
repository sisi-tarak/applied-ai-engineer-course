# Module 01 — What Is Applied AI Engineering?

> **Duration:** 1 hour | **Difficulty:** Beginner | **Prerequisites:** None

**[← Back to Main](../../README.md)**

---

## The Big Picture

```
THE AI ECOSYSTEM — WHERE YOU FIT:

┌─────────────────────────────────────────────────────────┐
│                   AI RESEARCH TEAM                       │
│   (OpenAI, Anthropic, Google DeepMind, Meta AI)         │
│   Build foundation models: GPT-4, Claude, Gemini        │
│   Cost: $100M–$500M | Team: 100+ PhDs                   │
└──────────────────────────┬──────────────────────────────┘
                           │ releases model via API
                           ▼
┌─────────────────────────────────────────────────────────┐
│              YOU — APPLIED AI ENGINEER                   │
│   Take foundation models → Customize for businesses     │
│   Build: RAG systems, fine-tuned models, AI agents      │
│   Cost: $500–$50,000 | Team: 1–5 engineers              │
│   Salary: ₹10–80 LPA                                    │
└──────────────────────────┬──────────────────────────────┘
                           │ delivers AI product
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    BUSINESSES                            │
│   Hospital, Law firm, Bank, E-commerce, HR team         │
│   Use the AI product you built                          │
│   Pay: ₹5L–₹1Cr per project                            │
└─────────────────────────────────────────────────────────┘
```

---

## What Applied AI Engineers Actually Build

### Type 1 — RAG Systems (Most Common — 89% of jobs)

```
PROBLEM: Company has 50,000 internal documents.
         Employees waste hours searching for answers.

YOUR SOLUTION:
  Documents → Chunked → Embedded → Stored in VectorDB
                                          ↓
  Employee asks question → Similar chunks retrieved
                                          ↓
  LLM reads chunks + question → Accurate answer

REAL EXAMPLES:
  - Hospital chatbot that knows all drug interactions
  - Legal firm assistant that searches 10,000 case files
  - HR bot that answers leave policy questions instantly
  - Bank compliance checker that reads regulations
```

### Type 2 — Fine-tuned Models (High-value — ₹40L+ roles)

```
PROBLEM: GPT-4 is good at English. 
         A Tamil newspaper needs AI that writes perfect Tamil news.

YOUR SOLUTION:
  Take Llama 4 (open source, free) →
  Fine-tune on 100,000 Tamil news articles →
  Result: Model that writes Tamil like a native journalist

REAL EXAMPLES:
  - Medical diagnosis assistant (fine-tuned on patient data)
  - Legal document drafter (fine-tuned on Indian law)
  - Customer support bot (fine-tuned on company's past chats)
  - Code review AI (fine-tuned on company's codebase)
```

### Type 3 — AI Agents (Fastest growing — automation focus)

```
PROBLEM: HR team manually screens 500 resumes per month.
         Takes 2 weeks. Error-prone. Expensive.

YOUR SOLUTION:
  Resume Scanner Agent:
    Tool 1: Extract skills from resume
    Tool 2: Compare against job requirements  
    Tool 3: Score candidate 1-100
    Tool 4: Draft personalised rejection/acceptance email
    Tool 5: Update Applicant Tracking System
  
  Agent runs autonomously → 500 resumes screened in 2 hours

REAL EXAMPLES:
  - Invoice processing automation
  - Social media content generation pipeline
  - Competitor monitoring agent
  - Customer onboarding automation
```

### Type 4 — Multi-modal AI (Emerging — cutting edge)

```
PROBLEM: Insurance company receives 1000 claim photos/day.
         Manual review costs ₹50/photo, takes 3 days.

YOUR SOLUTION:
  Photo → Vision LLM → Damage assessment
                    → Fraud detection
                    → Claim amount estimation
                    → Decision in 30 seconds
                    → Human reviews only flagged cases

REAL EXAMPLES:
  - Product defect detection in manufacturing
  - Medical image analysis (X-ray, MRI)
  - Document digitization (handwritten forms → structured data)
  - Video surveillance with AI alerts
```

---

## The Three Customization Techniques — Decision Guide

This is the most important diagram in this entire course:

```
                    NEW BUSINESS PROBLEM
                           │
                           ▼
              ┌────────────────────────┐
              │ Does LLM already know  │
              │ how to answer this     │
              │ with better prompting? │
              └────────────────────────┘
                    │           │
                   YES          NO
                    │           │
                    ▼           ▼
            ┌──────────┐  ┌────────────────────────┐
            │ PROMPT   │  │ Does company have       │
            │ ENGINEER │  │ private documents /     │
            │ IT        │  │ data the LLM needs?    │
            └──────────┘  └────────────────────────┘
              Module 3           │           │
                                YES          NO
                                 │           │
                                 ▼           ▼
                          ┌──────────┐  ┌──────────────────┐
                          │  BUILD   │  │ Does company need │
                          │  A RAG   │  │ specific style /  │
                          │  SYSTEM  │  │ tone / format?    │
                          └──────────┘  └──────────────────┘
                            Module 4          │          │
                                            YES          NO
                                              │          │
                                              ▼          ▼
                                       ┌──────────┐  ┌──────────┐
                                       │ FINE     │  │ RAG is   │
                                       │ TUNE A   │  │ probably │
                                       │ MODEL    │  │ enough   │
                                       └──────────┘  └──────────┘
                                         Module 5
```

**Rule of thumb:**
- Try prompting first — cheapest, fastest, often sufficient
- Add RAG when the LLM needs private/updated information
- Fine-tune only when style, tone, or specialized behavior is needed
- Use agents when the task requires multiple steps and tools

---

## Day in the Life — Applied AI Engineer

```
9:00 AM  → Client call: Hospital wants to reduce admin work
9:30 AM  → Scope the solution: RAG + fine-tuned notes generator
10:00 AM → Set up development environment
10:30 AM → Load hospital documents into ChromaDB (RAG phase 1)
12:00 PM → Test retrieval quality — find chunking issues
1:00 PM  → Lunch break
2:00 PM  → Fix chunking strategy (semantic vs fixed-size)
3:00 PM  → Evaluate RAG with RAGAS framework (F1, faithfulness)
4:00 PM  → Start fine-tuning Llama on clinical note samples
5:00 PM  → Set up training job on RunPod GPU cloud
6:00 PM  → Demo with 3 test cases — 90% accuracy achieved
6:30 PM  → Push to staging server via FastAPI
7:00 PM  → Send update to client with demo video
```

---

## Salary & Career Path

```
ENTRY LEVEL (0–2 years):
  Title: Junior AI Engineer / AI Developer
  Salary: ₹10–18 LPA
  Skill: RAG systems + basic fine-tuning + deployment

MID LEVEL (2–4 years):
  Title: Applied AI Engineer / LLM Engineer
  Salary: ₹18–40 LPA
  Skill: Production RAG + LoRA/QLoRA + agent systems

SENIOR LEVEL (4+ years):
  Title: Senior AI Engineer / AI Architect
  Salary: ₹40–80 LPA
  Skill: Full stack AI — custom infra + evals + team lead

FREELANCE / CONSULTING:
  Entry: ₹2–5L per project
  Experienced: ₹8–25L per project
  Specialised: ₹25L–1Cr per project

COMPANIES HIRING IN INDIA (2026):
  AI Startups: Sarvam AI, Krutrim, Postman AI, Agentive
  Product: Freshworks, Zoho, Razorpay, PhonePe, Meesho
  MNC India: Microsoft, Google, Amazon, IBM, Accenture
  Consulting: Deloitte AI, McKinsey Tech, BCG Gamma
  Your own clients as freelancer
```

---

## What This Course Will Not Teach

Being honest about scope prevents wasted time:

```
❌ Building GPT-4 from scratch
   (Requires $500M, 10,000 GPUs, 500 PhDs, 3 years)

❌ Deep ML mathematics research
   (Good career, but different path — takes a PhD)

❌ Data Science (analytics, dashboards, statistics)
   (Adjacent but different role)

❌ MLOps at hyperscale (Netflix/Google level)
   (Requires 5+ years of production experience)

✅ WHAT THIS COURSE TEACHES:
   Building real AI products companies pay for
   Using the best tools available today
   Getting hired as an Applied AI Engineer
```

---

## Resources for This Module

| Resource | Type | Time | Why |
|----------|------|------|-----|
| [AI Engineer vs ML Engineer — KORE1](https://www.kore1.com/llm-engineer-vs-ml-engineer/) | 📖 Blog | 15 min | Best explanation of the distinction |
| [What is RAG — IBM](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) | 📖 Blog | 10 min | Official RAG explanation |
| [Applied AI Engineer role — LinkedIn Jobs](https://www.linkedin.com/jobs/search/?keywords=applied+ai+engineer) | 🔍 Research | 20 min | Read 10 real job descriptions |
| [State of AI Report 2025 — Air Street](https://www.stateof.ai/) | 📊 Report | 1 hr | Market landscape |
| [LLM Engineer Roadmap — roadmap.sh](https://roadmap.sh/ai-engineer) | 🗺️ Visual | 10 min | Visual skill map |

---

**[← Back to Main](../../README.md) | [Next: Module 2 — Python Setup →](../02-python-for-ai/README.md)**
