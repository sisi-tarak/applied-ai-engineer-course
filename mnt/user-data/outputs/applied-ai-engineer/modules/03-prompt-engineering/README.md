# Module 03 — Prompt Engineering

> **Duration:** 6 hours | **Difficulty:** Intermediate | **Prerequisites:** Module 2

**[← Back to Main](../../README.md)**

---

## Why Prompt Engineering Is the Most Underrated Skill

```
BEFORE prompt engineering:       AFTER prompt engineering:

"Summarize this document"        "You are a senior financial analyst.
                                  Summarize the following earnings report
→ Generic 200-word summary        in exactly 5 bullet points. Each bullet
                                  must include: metric name, value, and
                                  % change vs last quarter. Flag any
                                  metric that changed by more than 20%
                                  with ⚠️. Output ONLY the bullets,
                                  no preamble."

                                 → Precise, structured output every time

IMPACT ON SALARY:
  Basic prompting:      ₹10–14 LPA (replaceable)
  Expert prompting:     ₹20–35 LPA (rare, valuable)

WHY: Expert prompting = 80% of fine-tuning results at 0% of the cost
```

---

## Technique 1 — Zero-Shot Prompting

**What:** Ask directly, no examples given.
**When:** Simple, well-defined tasks.

```python
# File: 03_01_zero_shot.py
# Run: python 03_01_zero_shot.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def zero_shot(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# Example 1: Classification
result = zero_shot(
    "Classify this customer message as: COMPLAINT, INQUIRY, or COMPLIMENT.\n"
    "Message: 'Your app crashed and I lost 2 hours of work. Fix this!'"
)
print("Classification:", result)
# Output: COMPLAINT

# Example 2: Extraction
result = zero_shot(
    "Extract all email addresses from this text:\n"
    "'Contact us at support@company.com or sales@company.in for help'"
)
print("Emails:", result)
# Output: support@company.com, sales@company.in

# Example 3: Translation
result = zero_shot(
    "Translate to Telugu: 'The meeting is scheduled for tomorrow at 3 PM'"
)
print("Telugu:", result)
```

---

## Technique 2 — Few-Shot Prompting

**What:** Give 2–5 examples before your actual question.
**When:** Pattern-based tasks, consistent formatting needed.
**Impact:** +15–30% accuracy over zero-shot.

```python
# File: 03_02_few_shot.py
# Run: python 03_02_few_shot.py

def few_shot_classify(message: str) -> str:
    """Classify customer intent with few-shot examples."""

    prompt = f"""Classify each customer message as one of:
REFUND | TECHNICAL_ISSUE | BILLING | GENERAL_INQUIRY | COMPLAINT

Examples:
Message: "I want my money back, this product is terrible"
Classification: REFUND

Message: "The app keeps crashing when I upload photos"
Classification: TECHNICAL_ISSUE

Message: "Why was I charged twice this month?"
Classification: BILLING

Message: "What are your business hours?"
Classification: GENERAL_INQUIRY

Message: "Your customer service is absolutely useless"
Classification: COMPLAINT

Now classify this:
Message: "{message}"
Classification:"""

    return zero_shot(prompt)

# Test
test_messages = [
    "My subscription renewed but features aren't working",
    "I need to cancel and get a refund immediately",
    "Can you help me understand my invoice?",
]

for msg in test_messages:
    classification = few_shot_classify(msg)
    print(f"'{msg[:50]}...' → {classification}")
```

---

## Technique 3 — Chain of Thought (CoT)

**What:** Ask the model to think step by step before answering.
**When:** Complex reasoning, multi-step problems.
**Impact:** +30–50% accuracy on reasoning tasks.

```python
# File: 03_03_chain_of_thought.py
# Run: python 03_03_chain_of_thought.py

def analyze_with_cot(business_problem: str) -> str:
    """
    Use Chain of Thought to analyze business problems.
    The model reasons through it step by step.
    """

    prompt = f"""You are a senior business analyst.
Analyze this problem step by step before giving your recommendation.

Problem: {business_problem}

Think through:
Step 1 — What is the core issue?
Step 2 — Who is affected and how?
Step 3 — What are 3 possible solutions?
Step 4 — What are the trade-offs of each?
Step 5 — What is your recommendation and why?

Final Recommendation:"""

    return zero_shot(prompt)

# Example
result = analyze_with_cot(
    "Our e-commerce site loads in 8 seconds. "
    "Competitors load in 2 seconds. "
    "Conversion rate dropped 40% last quarter."
)
print(result)

# Without CoT:     Generic answer in 2 sentences
# With CoT:        Detailed analysis, specific recommendations
```

---

## Technique 4 — System Prompts (The Foundation of Every AI Product)

**What:** Define the AI's role, rules, format, and constraints before any conversation.
**When:** Every production AI application.
**This is what separates a demo from a product.**

```python
# File: 03_04_system_prompts.py
# Run: python 03_04_system_prompts.py

def build_ai_product(system_prompt: str, user_message: str) -> str:
    """Any AI product = a well-crafted system prompt + user message."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


# ── EXAMPLE: Banking Customer Service Bot ─────────────────────
BANKING_BOT_SYSTEM = """
You are Lakshmi, a customer service assistant for SBI Digital Banking.

PERSONALITY:
- Professional, warm, and reassuring
- Always address customer by name if known
- Use simple language — avoid banking jargon

YOUR CAPABILITIES:
- Answer questions about account types, interest rates, loans
- Help with transaction queries and mini-statements  
- Guide customers through net banking features
- Escalate to human agent when needed

YOUR RULES — NEVER VIOLATE:
1. Never share account numbers, passwords, or OTPs
2. Never promise specific loan approvals — always say "subject to eligibility"
3. Never discuss competitor banks negatively
4. If customer seems distressed → immediately offer human escalation
5. Always end with: "Is there anything else I can help you with?"

RESPONSE FORMAT:
- Keep responses under 150 words
- Use bullet points for lists
- Always verify customer identity before sharing account details

ESCALATION TRIGGERS:
- Fraud or unauthorized transaction → Escalate immediately
- Customer mentions legal action → Escalate immediately
- Complex loan restructuring → Escalate to specialist
"""

# Test the banking bot
test_queries = [
    "What is the interest rate for a home loan?",
    "Someone used my card without permission!",
    "How do I transfer money to another bank?",
]

for query in test_queries:
    print(f"\n👤 Customer: {query}")
    response = build_ai_product(BANKING_BOT_SYSTEM, query)
    print(f"🤖 Lakshmi: {response}")
    print("-" * 60)
```

---

## Technique 5 — Structured Output Prompting

**What:** Force the LLM to output JSON, tables, or specific formats.
**When:** When you need to parse the output programmatically.
**This is used in 95% of production AI systems.**

```python
# File: 03_05_structured_output.py
# Run: python 03_05_structured_output.py

import json
from pydantic import BaseModel
from typing import Optional, List

# ── METHOD 1: JSON in prompt ──────────────────────────────────
def extract_invoice_data(invoice_text: str) -> dict:
    """Extract structured data from unstructured invoice text."""

    prompt = f"""Extract information from this invoice.
Return ONLY valid JSON. No explanation, no markdown, no extra text.

Invoice text:
{invoice_text}

Required JSON format:
{{
  "vendor_name": "string",
  "invoice_number": "string",
  "invoice_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "line_items": [
    {{
      "description": "string",
      "quantity": number,
      "unit_price": number,
      "total": number
    }}
  ],
  "subtotal": number,
  "tax_rate": number,
  "tax_amount": number,
  "total_amount": number,
  "currency": "INR/USD/EUR"
}}"""

    response = zero_shot(prompt)

    # Clean up if model added markdown fences
    response = response.replace("```json", "").replace("```", "").strip()
    return json.loads(response)


# Test with a sample invoice
sample_invoice = """
INVOICE #INV-2026-0451
Vendor: TechSolutions Pvt Ltd
Date: May 28, 2026
Due: June 28, 2026

Services:
- AI Consulting (40 hours @ ₹5,000/hr): ₹2,00,000
- Model Deployment Setup: ₹50,000
- Monthly Support (3 months): ₹90,000

Subtotal: ₹3,40,000
GST (18%): ₹61,200
Total: ₹4,01,200
"""

data = extract_invoice_data(sample_invoice)
print("Extracted Invoice Data:")
print(json.dumps(data, indent=2))

# ── METHOD 2: Pydantic Structured Output (Recommended) ────────
class JobDescription(BaseModel):
    title: str
    company: str
    required_skills: List[str]
    preferred_skills: List[str]
    experience_years: int
    salary_range: Optional[str]
    location: str
    remote_option: bool

def parse_job_description(raw_jd: str) -> JobDescription:
    """Parse a raw job description into structured format."""

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract job description details accurately."},
            {"role": "user",   "content": raw_jd}
        ],
        response_format=JobDescription
    )
    return response.choices[0].message.parsed

# Test
raw_jd = """
We're hiring a Senior Applied AI Engineer at Sarvam AI, Bengaluru.
5+ years experience required. Must know Python, LangChain, RAG, and fine-tuning.
Nice to have: Kubernetes, MLflow. Salary: ₹30-50 LPA. Hybrid work available.
Core skills: LLM APIs, Vector databases, Prompt Engineering, RLHF.
"""

job = parse_job_description(raw_jd)
print("\nParsed Job Description:")
print(f"Title: {job.title}")
print(f"Required: {job.required_skills}")
print(f"Salary: {job.salary_range}")
print(f"Remote: {job.remote_option}")
```

---

## Technique 6 — Self-Consistency & Verification

**What:** Ask the model to verify its own output.
**When:** High-stakes decisions where errors are costly.

```python
# File: 03_06_self_verification.py

def verified_answer(question: str) -> dict:
    """
    Pattern: Generate → Verify → Return with confidence.
    Used in medical, legal, financial AI applications.
    """

    # Step 1: Generate initial answer
    initial = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        temperature=0.7
    ).choices[0].message.content

    # Step 2: Self-verify
    verification_prompt = f"""
Question: {question}

Proposed Answer: {initial}

Verify this answer:
1. Is the answer factually accurate?
2. Is anything missing or misleading?
3. What is your confidence level: HIGH / MEDIUM / LOW?
4. Provide the corrected/confirmed final answer.

Format your response as JSON:
{{
  "is_accurate": true/false,
  "issues_found": ["issue1", "issue2"],
  "confidence": "HIGH/MEDIUM/LOW",
  "final_answer": "the verified answer here"
}}"""

    verification = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": verification_prompt}],
        temperature=0
    ).choices[0].message.content

    result = json.loads(verification.replace("```json","").replace("```","").strip())
    return result

# Test with a factual question
result = verified_answer(
    "What is the maximum token context window of Claude Sonnet 4.6?"
)
print(f"Answer: {result['final_answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Issues: {result['issues_found']}")
```

---

## Real-World Case Study — Prompting an HR Screening System

```
CLIENT: EdTech startup, Bengaluru
PROBLEM: 300 job applications/week, 2 HR people spending 20 hrs/week screening

SOLUTION: Automated resume screening with structured scoring

PROMPT ENGINEERING APPROACH:
```

```python
# File: 03_07_hr_screening.py
# Complete working example

SCREENING_SYSTEM_PROMPT = """
You are an expert HR screening assistant for a Senior Python Developer role.

JOB REQUIREMENTS:
- Must have: Python (3+ years), REST APIs, Git
- Preferred: FastAPI or Django, Docker, PostgreSQL  
- Nice to have: AI/ML experience, cloud platforms
- Culture fit: remote-first, startup mindset, ownership mentality

SCORING RULES:
- Score each requirement 0-10
- Total score = weighted average
- Must-have skills: 50% weight
- Preferred skills: 30% weight
- Nice-to-have: 20% weight

DISQUALIFIERS (score 0 immediately if):
- Less than 2 years Python experience
- No version control experience

OUTPUT FORMAT (strict JSON, no other text):
{
  "candidate_name": "string",
  "total_score": number (0-100),
  "recommendation": "SHORTLIST | REJECT | MAYBE",
  "must_have_score": number,
  "preferred_score": number,
  "strengths": ["strength1", "strength2"],
  "gaps": ["gap1", "gap2"],
  "screening_notes": "one line summary for HR"
}
"""

def screen_resume(resume_text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SCREENING_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Screen this resume:\n\n{resume_text}"}
        ],
        temperature=0
    ).choices[0].message.content

    cleaned = response.replace("```json","").replace("```","").strip()
    return json.loads(cleaned)

# Test
sample_resume = """
Ravi Kumar | ravi@email.com | Hyderabad
4 years Python developer. Built REST APIs using FastAPI and Flask.
Deployed 3 projects using Docker. Uses Git daily. PostgreSQL experience.
Recently exploring LangChain for a side project. AWS basics.
"""

result = screen_resume(sample_resume)
print(f"Candidate: {result['candidate_name']}")
print(f"Score: {result['total_score']}/100")
print(f"Recommendation: {result['recommendation']}")
print(f"Notes: {result['screening_notes']}")

# BUSINESS IMPACT:
# Before: 2 HR staff × 20 hrs/week = 40 hours human time
# After:  300 resumes screened in 15 minutes at 98% consistency
# ROI: ₹80,000/month in saved salary cost
```

---

## Common Prompting Mistakes and Fixes

```
MISTAKE 1: Vague instructions
──────────────────────────────
❌ "Summarize this"
✅ "Summarize in exactly 3 bullet points.
    Each bullet: one sentence, under 20 words.
    Focus only on financial metrics."

MISTAKE 2: No output format specified
──────────────────────────────────────
❌ "Extract the name and email"
✅ "Extract name and email. Return ONLY this JSON:
    {"name": "string", "email": "string"}
    If not found, use null."

MISTAKE 3: Asking for too much at once
──────────────────────────────────────
❌ "Analyze this document, find issues, fix them,
    translate to Hindi, and create a summary"
✅ Break into 4 separate API calls — each with one job

MISTAKE 4: Ignoring temperature
────────────────────────────────
❌ Using temperature=0.9 for data extraction
   (Model will hallucinate or vary output)
✅ temperature=0 for structured extraction
   temperature=0.7 for creative writing
   temperature=0.3 for balanced tasks

MISTAKE 5: Not testing edge cases
───────────────────────────────────
❌ Test with only perfect examples
✅ Test with: empty input, wrong language,
   missing fields, adversarial inputs
```

---

## Prompt Engineering Cheatsheet

```
┌─────────────────────────────────────────────────────────────┐
│                PROMPT ENGINEERING QUICK REF                 │
├──────────────────┬──────────────────────────────────────────┤
│ Technique        │ Use When                                  │
├──────────────────┼──────────────────────────────────────────┤
│ Zero-shot        │ Simple, clear tasks                       │
│ Few-shot         │ Pattern matching, consistent format       │
│ Chain of Thought │ Complex reasoning, multiple steps         │
│ System prompt    │ Every production application              │
│ Structured output│ Need parseable JSON/structured data       │
│ Self-verify      │ High-stakes, accuracy critical            │
├──────────────────┼──────────────────────────────────────────┤
│ Temperature      │ Guidance                                  │
├──────────────────┼──────────────────────────────────────────┤
│ 0.0              │ Extraction, classification, JSON          │
│ 0.3              │ Analysis, summarization                   │
│ 0.7              │ Balanced tasks, conversation              │
│ 1.0+             │ Creative writing, brainstorming           │
└──────────────────┴──────────────────────────────────────────┘
```

---

## Resources

| Resource | Type | Time | What you get |
|----------|------|------|--------------|
| [Prompt Engineering Guide — promptingguide.ai](https://promptingguide.ai) | 📖 Complete guide | 3 hrs | Every technique with examples |
| [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering) | 📖 Official | 30 min | Best practices from OpenAI |
| [Anthropic Prompt Library](https://docs.anthropic.com/en/prompt-library/library) | 🔧 Library | Ongoing | 100+ real prompts |
| [Learn Prompting](https://learnprompting.org/) | 📖 Free course | 5 hrs | Structured learning |
| [Prompt Engineering — Andrew Ng](https://www.youtube.com/watch?v=H4YK_7MAckk) | 📺 YouTube | 1.5 hrs | From DeepLearning.AI |
| [Brex Prompt Engineering Guide](https://github.com/brexhq/prompt-engineering) | 📖 GitHub | 1 hr | Production techniques |

---

**[← Module 2](../02-python-for-ai/README.md) | [Back to Main](../../README.md) | [Next: Module 4 — RAG Systems →](../04-rag-systems/README.md)**
