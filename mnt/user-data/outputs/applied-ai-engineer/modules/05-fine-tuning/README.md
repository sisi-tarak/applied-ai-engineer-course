# Module 05 — Fine-tuning LLMs with LoRA & QLoRA

> **Duration:** 8 hours | **Difficulty:** Advanced | **Prerequisites:** Module 4

**[← Back to Main](../../README.md)**

---

## Fine-tuning vs RAG — The Decision That Changes Your Salary

```
                    WHEN TO USE EACH:

RAG:                              Fine-tuning:
──────────────────────────────    ──────────────────────────────
Company needs current data        Company needs specific STYLE
Data changes frequently           Behavior must be consistent
Multiple data sources             Domain vocabulary is specialized
Fast to implement (days)          Willing to invest (weeks)
₹3L–8L project                   ₹8L–25L project

EXAMPLE: Legal firm chatbot
  RAG answer:    "Based on the Sections uploaded, the limitation
                  period for contracts is 3 years..."
  Fine-tuned:    "Per the Indian Contract Act 1872, Section 14,
                  the limitation period... [formal legal tone,
                  specific citations, court-ready language]"

The difference: RAG retrieves facts. Fine-tuning changes HOW it speaks.
```

---

## The Fine-tuning Landscape (2026)

```
METHOD          VRAM NEEDED    QUALITY    COST          USE CASE
──────────────────────────────────────────────────────────────────
Full fine-tune  80GB+ (A100)   Best       $$$$$         Research labs
LoRA            16–24GB        Very good  $$            Production apps
QLoRA           8–12GB         Good       $             Consumer GPU
PEFT/Adapters   8GB            Good       $             Resource limited

For Applied AI Engineers: QLoRA is the standard
  → Fine-tune Llama 4 8B on a single RTX 4090 (₹1.2L GPU)
  → Or use cloud: RunPod, Vast.ai, Google Colab Pro (~₹800/hour)
```

---

## Understanding LoRA

```
FULL FINE-TUNING:
  Model has W = original weight matrix (billions of parameters)
  Update EVERY weight: W_new = W + ΔW
  Problem: ΔW is as large as W → needs huge memory

LORA (Low-Rank Adaptation):
  Instead of updating W directly:
  W_new = W + B × A
  Where:
    B = tall thin matrix (d × r)
    A = short wide matrix (r × k)
    r = rank (typically 8, 16, or 64)
    r << d and r << k

  EXAMPLE:
    W is 4096 × 4096 = 16.7M parameters
    With rank=16:
      B = 4096 × 16 = 65,536 parameters
      A = 16 × 4096 = 65,536 parameters
      Total LoRA: 131,072 parameters
    
    Memory saving: 16.7M vs 131K = 128x smaller!
    Quality loss: minimal (< 1% on most tasks)

QLORA = LoRA + 4-bit quantization:
  Further reduces base model from float32 (32-bit) to int4 (4-bit)
  Memory: 4x reduction + LoRA overhead
  Result: Fine-tune 7B model on 8GB GPU (RTX 3070 / M1 Mac!)
```

---

## Complete Fine-tuning Pipeline

```python
# File: 05_01_finetuning_pipeline.py
# Required: pip install transformers peft trl bitsandbytes datasets
# GPU: 8GB+ VRAM recommended. Or use Google Colab Pro.

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset

print("="*60)
print("FINE-TUNING PIPELINE")
print("="*60)

# ── STEP 1: Configure QLoRA (4-bit quantization) ──────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,               # Load model in 4-bit
    bnb_4bit_use_double_quant=True,  # Extra memory saving
    bnb_4bit_quant_type="nf4",       # NormalFloat4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute in bf16
)

# ── STEP 2: Load base model ───────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# Alternatives: "mistralai/Mistral-7B-Instruct-v0.3"
#               "Qwen/Qwen2.5-7B-Instruct"
#               "google/gemma-2-2b-it"
# All free on HuggingFace — just need an account + accept terms

print(f"Loading {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",              # Automatically use GPU
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Model loaded. Parameters: {model.num_parameters():,}")

# ── STEP 3: Configure LoRA ─────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # LoRA rank — higher = more params = better quality
    lora_alpha=32,           # Scaling factor (usually 2 × r)
    target_modules=[         # Which layers to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"        # MLP layers
    ],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 20,447,232 || all params: 3,252,756,480
# Only 0.63% of params are trainable! Everything else is frozen.

# ── STEP 4: Prepare Training Data ─────────────────────────────
# Format: instruction tuning (most common for business AI)
# Each example = a conversation the model should learn from

def format_example(system: str, instruction: str, response: str) -> str:
    """Format using Llama chat template."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}<|eot_id|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{response}<|eot_id|>"""

# Example: Fine-tuning for a medical AI assistant
SYSTEM_PROMPT = """You are a clinical documentation assistant for Apollo Hospitals.
You help doctors draft SOAP notes from bullet-point inputs.
Always use proper medical terminology and formal clinical language."""

training_examples = [
    {
        "text": format_example(
            SYSTEM_PROMPT,
            """Patient: 45M, presenting with chest pain 2/10 for 3 days.
            Exam: BP 140/90, HR 88, clear lungs, no JVD.""",
            """SOAP NOTE:
S: Mr. [Patient], a 45-year-old male, presents with a chief complaint of chest pain rated 2/10 on the pain scale, persistent for 3 days. Pain is non-radiating and the patient denies associated dyspnea, diaphoresis, or palpitations.

O: Vital signs: BP 140/90 mmHg, HR 88 bpm, RR 16/min, SpO2 98% on room air. Physical examination reveals clear lung fields bilaterally on auscultation. No jugular venous distension noted. Cardiac auscultation pending.

A: Chest pain, etiology to be determined. Hypertension noted. Rule out: (1) Musculoskeletal, (2) GERD, (3) ACS given hypertensive status.

P: 12-lead ECG, cardiac enzymes, lipid panel, comprehensive metabolic panel. Nitroglycerin PRN for chest pain. BP management: Amlodipine 5mg OD. Follow-up in 48 hours or sooner if symptoms worsen."""
        )
    },
    # Add 50–500 more examples for production quality
]

dataset = Dataset.from_list(training_examples)
print(f"Training dataset: {len(dataset)} examples")

# ── STEP 5: Training Configuration ─────────────────────────────
training_args = TrainingArguments(
    output_dir="./fine_tuned_medical_ai",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # Effective batch size = 16
    learning_rate=2e-4,
    fp16=False,
    bf16=True,                        # Better than fp16 on modern GPUs
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="none",                 # Change to "wandb" for tracking
    optim="paged_adamw_32bit",        # Memory-efficient optimizer
)

# ── STEP 6: Train ──────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
)

print("Starting training...")
# trainer.train()   # Uncomment when ready to train
print("Training complete!")

# ── STEP 7: Save and Merge ─────────────────────────────────────
# trainer.model.save_pretrained("./medical_ai_lora")
# tokenizer.save_pretrained("./medical_ai_lora")

# For production: merge LoRA weights into base model
# from peft import PeftModel
# merged_model = PeftModel.from_pretrained(base_model, "./medical_ai_lora")
# merged_model = merged_model.merge_and_unload()
# merged_model.save_pretrained("./medical_ai_merged")
```

---

## Data Preparation — The Most Critical Step

```python
# File: 05_02_data_preparation.py
# Quality of training data > everything else

import json
import random
from pathlib import Path

def create_instruction_dataset(
    raw_examples: list,
    output_file: str,
    system_prompt: str,
    train_split: float = 0.9
) -> dict:
    """
    Convert raw examples into instruction-tuning format.
    Includes quality checks and train/val split.
    """

    formatted = []
    rejected  = []

    for example in raw_examples:
        instruction = example.get("input", "")
        response    = example.get("output", "")

        # Quality checks
        if len(instruction) < 20:
            rejected.append({"reason": "instruction too short", "example": example})
            continue
        if len(response) < 50:
            rejected.append({"reason": "response too short", "example": example})
            continue
        if len(response) > 4000:
            rejected.append({"reason": "response too long", "example": example})
            continue

        formatted.append({
            "messages": [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": instruction},
                {"role": "assistant", "content": response}
            ]
        })

    # Shuffle and split
    random.shuffle(formatted)
    split_idx  = int(len(formatted) * train_split)
    train_data = formatted[:split_idx]
    val_data   = formatted[split_idx:]

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    with open(f"{output_file}_train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(f"{output_file}_val.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset created:")
    print(f"  Total examples: {len(formatted)}")
    print(f"  Rejected:       {len(rejected)}")
    print(f"  Training set:   {len(train_data)}")
    print(f"  Validation set: {len(val_data)}")

    return {"train": train_data, "val": val_data, "rejected": rejected}


# HOW MANY EXAMPLES DO YOU NEED?
print("""
TRAINING DATA REQUIREMENTS:

Task Complexity         Min Examples   Ideal     Notes
────────────────────────────────────────────────────────
Simple classification   50–100        500+      Format/style
Named entity extract    200–500       1000+     Consistent labels
Domain adaptation       500–1000      2000+     Industry vocab
Full task learning      1000–5000     10,000+   New capabilities
Instruction following   2000+         10,000+   General purpose

DATA SOURCES (where to get training data):
1. Your own past work
   - Customer support chat logs (anonymized)
   - Past documents created by subject experts
   - Corrected model outputs

2. Synthetic data generation
   - Use GPT-4 to generate examples from your documents
   - Manual curation afterward: keep only high quality

3. Public datasets (domain-specific)
   - HuggingFace Datasets: https://huggingface.co/datasets
   - Kaggle: https://kaggle.com/datasets
   - Government open data portals

DATA QUALITY > DATA QUANTITY:
  100 perfect examples >> 1000 mediocre examples
  Always manually review 10% of your dataset before training
""")
```

---

## Running Fine-tuning on Cloud GPU

```bash
# Option 1: RunPod (recommended — cheapest)
# 1. Go to runpod.io
# 2. Deploy a pod: RTX 4090 (24GB VRAM) = $0.39/hour
# 3. Use this template: "RunPod PyTorch 2.1"
# 4. Upload your training script and data

# Option 2: Google Colab Pro
# $10/month → access to A100 GPU
# Enough for fine-tuning 7B models in 4-bit

# Option 3: Vast.ai (cheapest option)
# RTX 3090 = $0.15-0.25/hour
# Good for experiments, less reliable than RunPod

# Estimated costs for a 7B model, 1000 examples, 3 epochs:
# RunPod RTX 4090: ~2-3 hours = $0.80–$1.20
# Google Colab Pro: ~3-4 hours (A100) = included in subscription
```

---

## When NOT to Fine-tune

```
DO NOT fine-tune when:

1. You have < 100 examples → Use RAG or few-shot prompting
2. Data changes frequently → RAG is better (no retraining)
3. You need factual accuracy → RAG (fine-tuning can hallucinate)
4. Limited budget for GPU time → RAG is cheaper and faster
5. Quick prototype/demo → RAG first, fine-tune only if needed

FINE-TUNE WHEN:

1. Style must be consistent and domain-specific
2. Vocabulary is highly specialized (medical, legal, Telugu)
3. Model must refuse certain topics reliably
4. Task is repetitive and well-defined (extraction, formatting)
5. You have 500+ quality examples of the target behavior
6. Latency matters (fine-tuned local model vs API call)
```

---

## Resources

| Resource | Type | Time | What you get |
|----------|------|------|--------------|
| [Fine-tuning Llama — Unsloth](https://github.com/unslothAI/unsloth) | 🔧 Library | 1 hr | 5x faster fine-tuning |
| [QLoRA Paper](https://arxiv.org/abs/2305.14314) | 📄 Paper | 1 hr | Original technique |
| [HuggingFace PEFT Docs](https://huggingface.co/docs/peft) | 📖 Docs | 2 hrs | Official LoRA guide |
| [Fine-tuning Crash Course — Maxime Labonne](https://www.youtube.com/watch?v=pK6PTGrCZ7A) | 📺 YouTube | 2 hrs | Hands-on tutorial |
| [TRL SFTTrainer Docs](https://huggingface.co/docs/trl/sft_trainer) | 📖 Docs | 1 hr | Training library |
| [Alpaca Dataset Format](https://github.com/tatsu-lab/stanford_alpaca) | 📖 GitHub | 30 min | Standard data format |
| [RunPod Tutorial](https://www.youtube.com/watch?v=V7DzHK_mGRo) | 📺 YouTube | 30 min | Cloud GPU setup |

---

**[← Module 4](../04-rag-systems/README.md) | [Back to Main](../../README.md) | [Next: Module 6 — Agents →](../06-agents/README.md)**
