# Contributing to Applied AI Engineer

Thank you for helping make this course better for every engineer.

---

## What You Can Contribute

### 🐛 Fix Errors
Found a bug in code? Wrong explanation? Outdated API syntax?
Open an issue or submit a PR directly.

### 📝 Improve Explanations
Add clearer analogies. Improve Telugu translations. Simplify complex concepts.

### 💻 Add Code Examples
Add real-world examples from your industry:
- Healthcare AI? Add medical RAG examples
- Legal? Add document analysis examples
- Finance? Add trading/banking examples

### 🌍 Translate to Your Language
See [TRANSLATIONS.md](TRANSLATIONS.md) for guidelines.
Current priority:
- Hindi (`README-hi.md`) — 500M speakers
- Telugu (`README-te.md`) — 85M speakers
- Tamil (`README-ta.md`) — 80M speakers
- Kannada (`README-kn.md`) — 45M speakers
- Bengali (`README-bn.md`) — 230M speakers

### ❓ Add Interview Questions
Add real questions you were asked in Applied AI interviews.
Format: Question → Framework for answering → Sample answer.

### 📊 Add Case Studies
Add a real-world AI system you built or studied.
Format: Problem → Architecture → Results → Lessons.

---

## How to Submit a PR

```bash
# 1. Fork the repository
# (Click "Fork" button on GitHub)

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/applied-ai-engineer.git
cd applied-ai-engineer

# 3. Create a branch
git checkout -b add-healthcare-rag-example

# 4. Make your changes

# 5. Test your code changes (if applicable)
python -m pytest tests/

# 6. Commit with a clear message
git add .
git commit -m "Add: healthcare RAG example with HIPAA compliance notes"

# 7. Push to your fork
git push origin add-healthcare-rag-example

# 8. Open a Pull Request on GitHub
# Go to: github.com/sisi-tarak/applied-ai-engineer
# Click "New Pull Request"
```

---

## PR Guidelines

### ✅ Good PRs
- Fix a specific error with explanation
- Add a complete, runnable code example
- Add 5+ interview questions with model answers
- Add a full case study with architecture
- Translate a complete module

### ❌ PRs That Won't Be Merged
- Add links to paid courses
- Promote specific products without disclosure
- Unrelated to Applied AI Engineering
- Code that doesn't run
- Duplicate content already covered

---

## Code Standards

```python
# ✅ Good — includes explanation, docstring, type hints, example output
def chunk_document(text: str, chunk_size: int = 500) -> list[str]:
    """
    Split text into chunks for RAG indexing.

    Args:
        text: Input document text
        chunk_size: Target size per chunk in characters

    Returns:
        List of text chunks

    Example:
        >>> chunks = chunk_document("Long document...", chunk_size=100)
        >>> print(f"Created {len(chunks)} chunks")
        Created 3 chunks
    """
    ...

# ❌ Bad — no context, no types, no example
def chunk(t, s=500):
    ...
```

---

## Translation Guidelines

See [TRANSLATIONS.md](TRANSLATIONS.md) for full details.

Quick rules:
- Keep technical terms in English (RAG, LLM, fine-tuning, LoRA)
- Translate explanations naturally, not word-for-word
- Add a "(అర్థం: ...)" note when a concept is new to the language
- Mark status: `🚧 In Progress`, `✅ Complete`

---

## Contributors Hall of Fame

Every contributor gets:
- Your name and GitHub link in the Contributors section
- Credit on any content you substantially contributed
- Recognition in release notes

---

## Questions?

Open an issue with the label `question`.

Follow [@sisi_tarakk](https://instagram.com/sisi_tarakk) on Instagram for updates.
