"""
FangYuan-8B Hugging Face Hub Uploader
======================================
Uploads trained FangYuan-8B LoRA adapter weights, tokenizer, and an automated
rich Model Card (README.md) to the Hugging Face Hub.

Usage:
    py -3.10 push_to_hub.py
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables from .env
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
- xianxia
language:
- en
pipeline_tag: text-generation
library_name: peft
---

# ☯️ FangYuan-8B: Fang Yuan Persona & Cultivation Philosophy LLM

> *"I am laughing at myself, I am also laughing at all of you. Love and friendship, killing and slaughtering, don't you all find this very boring?"* — **Fang Yuan (方源)**

**FangYuan-8B** is a fine-tuned large language model engineered to embody the psychology, strategic calculation, and philosophical perseverance of **Fang Yuan (方源)**, the protagonist of the legendary web novel ***Reverend Insanity (蛊真人)*** by Gu Zhen Ren.

FangYuan-8B captures Fang Yuan's distinct cognitive framework:
* **Absolute Pragmatism & Utilitarianism**: Morals and institutions are evaluated purely through benefits vs. costs.
* **Stoic Tranquility & Anti-Regret**: Complete indifference to failure or death; the journey toward eternal life gives life meaning.
* **No Cartoon Villainy**: Calm, polite, and respectful on the outside; ruthlessly rational on the inside.
* **The Legends of Ren Zu**: Natural synthesis of allegorical parables (Hope Gu, Attitude Gu, Rules and Regulations).

---

## 📊 Dataset & Training Details

The model is fine-tuned on **4,901 high-density instruction-response pairs** synthesized directly from the full 2,334 chapters of *Reverend Insanity*.

* **Base Model**: [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)
* **Dataset Size**: 4,901 curated multi-turn dialogue pairs
* **Fine-Tuning Method**: QLoRA (4-bit NF4 Quantization)
* **LoRA Hyperparameters**:
  * Rank ($r$): `16`
  * Alpha ($\alpha$): `32`
  * Dropout: `0.0`
  * Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* **Schedule**: 1 Full Epoch with Cosine learning rate schedule

---

## 🏛️ The 5 Core Persona Archetypes

1. **Pragmatic Life Advice & Dilemmas**: Deconstructing societal conditioning, family loyalty, and morality as tools of control.
2. **The Legends of Ren Zu Interpretations**: Deep philosophical analysis of human nature, solitude, rules, and perseverance.
3. **Stoic Indifference & Anti-Regret**: Serenity in facing total annihilation, defeat, or betrayal.
4. **Machiavellian Tactical Scheming**: Navigating power structures, unassuming facades, and resource exploitation.
5. **In-Character Dialogue & Roleplay**: Polite, collected interactions with other cultivators and elders.

---

## 🚀 Quickstart & Inference

### 1. Installation

```bash
pip install torch transformers peft bitsandbytes accelerate
```

### 2. Python Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_REPO = "{repo_id}"

# 1. 4-bit Quantization Configuration
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

# 3. Attach Fang Yuan LoRA Adapter
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
print("Fang Yuan:\\n", response)
```

---

## 📜 Disclaimer
This model is a fan-created research and roleplay artifact exploring fictional novel philosophy. It is inspired by Gu Zhen Ren's *Reverend Insanity (蛊真人)*.
"""


def main():
    parser = argparse.ArgumentParser(description="Upload FangYuan-8B LoRA adapter to Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="lynzl/FangYuan-8B",
        help="Target Hugging Face repository ID (default: 'lynzl/FangYuan-8B')",
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
        help="Hugging Face access token (reads from .env by default)",
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
        return

    if not args.token:
        print("[-] Error: No Hugging Face token found. Please set HF_TOKEN in .env or pass --token.")
        return

    print("==========================================================")
    print("          FANGYUAN-8B HUGGING FACE HUB EXPORTER           ")
    print("==========================================================")
    print(f"Target HF Repository: {args.repo_id}")
    print(f"Adapter Directory:   {adapter_path}")

    # Generate Model Card README.md inside the adapter folder
    readme_path = adapter_path / "README.md"
    print(f"[+] Generating Hugging Face Model Card -> {readme_path}...")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(MODEL_CARD_TEMPLATE.format(repo_id=args.repo_id))

    # Initialize HF API
    api = HfApi(token=args.token)

    print(f"[+] Verifying repository on Hugging Face: {args.repo_id}...")
    try:
        create_repo(
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            exist_ok=True,
            repo_type="model",
        )
        print("    Repository verified / created successfully.")
    except Exception as e:
        print(f"[-] Notice: {e}")

    print(f"[+] Uploading all adapter files from '{adapter_path}' to '{args.repo_id}'...")
    try:
        upload_folder(
            folder_path=str(adapter_path),
            repo_id=args.repo_id,
            repo_type="model",
            token=args.token,
            commit_message="Upload FangYuan-8B LoRA adapter & Model Card",
        )
        print("\n==========================================================")
        print("SUCCESS! Model successfully published to Hugging Face Hub!")
        print(f"View your model live at: https://huggingface.co/{args.repo_id}")
        print("==========================================================")
    except Exception as e:
        print(f"\n[-] Upload failed: {e}")


if __name__ == "__main__":
    main()
