"""
FangYuan-8B Hugging Face Hub Uploader
======================================
Uploads your trained FangYuan-8B LoRA adapter weights, tokenizer, and an automated
rich Model Card (README.md) to the Hugging Face Hub.

Usage:
    python push_to_hub.py --repo_id <your-hf-username>/FangYuan-8B-LoRA
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: Qwen/Qwen3-8B
tags:
- lora
- peft
- roleplay
- reverend-insanity
- fang-yuan
- philosophy
- causal-lm
language:
- en
pipeline_tag: text-generation
library_name: peft
---

# FangYuan-8B: Fang Yuan Persona & Cultivation Philosophy LLM

> *"I am laughing at myself, I am also laughing at all of you. Love and friendship, killing and slaughtering, don't you all find this very boring?."* — **Fang Yuan**

**FangYuan-8B** is an instruction-tuned LLM fine-tuned to capture the philosophy and perseverance of **Fang Yuan (方源)**, the protagonist of the legendary web novel ***Reverend Insanity (蛊真人)*** by Gu Zhen Ren.

FangYuan-8B is designed from the ground up to embody Fang Yuan's distinct psychology, specifically his: **rationality, being completely unburdened by societal morality, possessing zero regrets, and analyzing life through calculations and allegorical wisdom.**

The model is trained on **4,901 curated instruction-response pairs** derived from all 2,334 chapters of *Reverend Insanity* using QLoRA 4-bit fine-tuning on **Qwen3-8B**.

---

## ✨ Core Archetypes

The model is fine-tuned on **4,901 high-density instruction-response pairs** structured across five distinct persona archetypes:

1. **Pragmatic Utilitarian**: Benefits over emotions; cost/benefit mindset.
2. **Legends of Ren Zu Wisdom**: Allegories on desire, solitude, rules, and hope.
3. **Stoic Indifference & Amor Fati**: Complete peace with ruin; "No regrets even in death", journey is the only reward.
4. **Tactical Scheming**: Exploiting clan rules; facade of mediocrity.
5. **Demonic Cultivation Roleplay**: Calm & polite externally; zero attachments.

---

## 🚀 Quickstart & Inference

### Installation

```bash
pip install torch transformers peft bitsandbytes accelerate
```

### Python Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_REPO = "{repo_id}"

# 1. Quantization configuration (4-bit NF4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

# 2. Load Tokenizer & Base Model
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 3. Load LoRA Adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
model.eval()

# 4. Generate with Fang Yuan's persona
system_prompt = "You are Fang Yuan, the protagonist of Reverend Insanity. You embody the Demonic Path—calm, rational, utilitarian, and utterly free of societal conditioning. You pursue Eternal Life with unyielding perseverance and zero regrets."
messages = [
    {{"role": "system", "content": system_prompt}},
    {{"role": "user", "content": "I worked hard for years at my company, but someone else got the promotion through connections. Should I be angry?"}}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.1,
    )

response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("Fang Yuan:", response)
```

---

## ⚙️ Training Details

- **Base Model**: `Qwen/Qwen3-8B`
- **Method**: QLoRA (4-bit NF4 Quantization + Paged 8-bit AdamW)
- **LoRA Hyperparameters**:
  - Rank ($r$): `16`
  - Alpha ($\\alpha$): `32`
  - Dropout: `0.05`
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Dataset Size**: 4,901 high-density instruction pairs
- **Epochs**: 1 Full Epoch
- **Learning Rate**: `2e-4` with Cosine decay
- **Hardware**: Fine-tuned on a single consumer GPU (RTX 4060 Ti 8GB VRAM)

---

## 📜 Disclaimer
This model is a fan-created research and roleplay artifact exploring fictional novel philosophy. It is inspired by Gu Zhen Ren's *Reverend Insanity*.
"""


def main():
    parser = argparse.ArgumentParser(description="Upload FangYuan-8B LoRA adapter to Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target Hugging Face repository ID (e.g. 'username/FangYuan-8B-LoRA')",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="./fangyuan_qwen3_8b_lora",
        help="Path to the trained LoRA adapter directory (default: './fangyuan_qwen3_8b_lora')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN", None),
        help="Hugging Face access token (or set HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private",
    )

    args = parser.parse_args()
    adapter_path = Path(args.adapter_dir).resolve()

    if not adapter_path.exists():
        print(f"[-] Error: Adapter directory '{adapter_path}' does not exist.")
        print("    Please run `py -3.10 train.py` first to train and save the adapter.")
        return

    print("==========================================================")
    print("          FANGYUAN-8B HUGGING FACE HUB EXPORTER           ")
    print("==========================================================")
    print(f"Target HF Repository: {args.repo_id}")
    print(f"Adapter Directory:   {adapter_path}")

    # Generate Model Card README.md inside the adapter folder
    readme_path = adapter_path / "README.md"
    print(f"[+] Writing Hugging Face Model Card to {readme_path}...")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(MODEL_CARD_TEMPLATE.format(repo_id=args.repo_id))

    # Initialize HF API
    api = HfApi(token=args.token)

    print(f"[+] Ensuring repository exists: {args.repo_id}...")
    try:
        create_repo(
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            exist_ok=True,
            repo_type="model",
        )
        print("    Repository ready.")
    except Exception as e:
        print(f"[-] Repository verification notice: {e}")

    print(f"[+] Uploading files from '{adapter_path}' to '{args.repo_id}'...")
    try:
        upload_folder(
            folder_path=str(adapter_path),
            repo_id=args.repo_id,
            repo_type="model",
            token=args.token,
            commit_message="Upload FangYuan-8B LoRA adapter & Model Card",
        )
        print("\n==========================================================")
        print("🎉 SUCCESS! Model successfully published to Hugging Face Hub!")
        print(f"🔗 View your model: https://huggingface.co/{args.repo_id}")
        print("==========================================================")
    except Exception as e:
        print(f"\n[-] Upload failed: {e}")
        print("    Tip: Check if your HF_TOKEN has WRITE permission or run `huggingface-cli login`.")


if __name__ == "__main__":
    main()
