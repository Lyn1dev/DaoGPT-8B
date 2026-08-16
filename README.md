# FangYuan-8B: Fang Yuan Persona & Cultivation Philosophy LLM

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/Base_Model-Qwen3--8B-purple.svg)](https://huggingface.co/Qwen/Qwen3-8B)
[![Fine-Tuning](https://img.shields.io/badge/Method-QLoRA_4--Bit_NF4-green.svg)](https://github.com/huggingface/peft)
[![Framework](https://img.shields.io/badge/TRL-SFTTrainer-orange.svg)](https://github.com/huggingface/trl)
[![Dataset](https://img.shields.io/badge/Dataset-4%2C901_Samples-blueviolet.svg)](data/sft_dataset.jsonl)

*“I am laughing at myself, I am also laughing at all of you. Love and friendship, killing and slaughtering, don't you all find this very boring?.”*

</div>

---

## 📌 Overview

**FangYuan-8B** is an instruction-tuned LLM fine-tuned to capture the philosophy and perseverance of **Fang Yuan (方源)**, the protagonist of the legendary web novel ***Reverend Insanity (蛊真人)*** by Gu Zhen Ren.

FangYuan-8B is designed from the ground up to embody Fang Yuan's distinct psychology, specifically his: **rationality, being completely unburdened by societal morality, possessing zero regrets, and analyzing life through calculations and allegorical wisdom.**

---

## ✨ Core Archetypes

The model is fine-tuned on **4,901 high-density instruction-response pairs** structured across five distinct persona archetypes:

1. Pragmatic Utilitarian: Benefits over emotions; cost/benefit mindset.

2. Legends of Ren Zu Wisdom: Allegories on desire, solitude, rules, and hope.

3. Stoic Indifference & Amor Fati: Complete peace with ruin; "No regrets even in death", journey is the only reward.

4. Tactical Scheming: Exploiting clan rules; facade of mediocrity.

5. Demonic Cultivation Roleplay: Calm & polite externally; zero attachments.
---

## 📂 Repository Structure

```
FangYuan-8B/
├── data/
│   ├── cleaner.py             # Rule-based cleaning pipeline (strips translator notes, footnotes, noise)
│   ├── extract.py             # EPUB extractor for all 2,334 novel chapters
│   └── sft_dataset.jsonl      # Curated 4,901 SFT pairs formatted in ChatML
├── generate_sft.py            # High-speed synthetic dataset generator (multi-threaded, persona-weighted)
├── train.py                   # QLoRA fine-tuning script optimized for 8GB VRAM (Qwen3-8B + NF4 + Paged AdamW)
├── test_model.py              # Interactive terminal CLI chat interface
├── push_to_hub.py             # Hugging Face Hub exporter & automated Model Card generator
├── requirements.txt           # Python dependency specifications
├── .env.example               # Template for API keys (HF_TOKEN, OPENROUTER_API_KEY)
├── .gitignore                 # Excludes weights, checkpoints, logs, and venvs
├── LICENSE                    # Apache 2.0 License
└── README.md                  # Project documentation
```

---

## 🛠️ Quickstart Guide

### 1. Prerequisites & Installation

Clone the repository and set up a Python 3.10+ virtual environment:

```bash
git clone https://github.com/Lyn1dev/FangYuan-8B.git
cd FangYuan-8B

# Create virtual environment
python -m venv daogpt-env
source daogpt-env/bin/activate  # On Windows: .\daogpt-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Fine-Tuning FangYuan-8B (`train.py`)

Run the fine-tuning script. It is pre-configured to run **1 full epoch** (~612 optimization steps) across all 4,901 samples with memory optimizations for consumer GPUs (e.g. RTX 4060 Ti 8GB VRAM):

```bash
python train.py
```

#### Training Specifications:
| Parameter | Setting |
|---|---|
| **Base Model** | `Qwen/Qwen3-8B` |
| **Quantization** | 4-bit NF4 (`bitsandbytes`) |
| **LoRA Rank / Alpha** | $r=16$, $\alpha=32$, Dropout = $0.05$ |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Optimizer** | `paged_adamw_8bit` |
| **Gradient Accumulation** | 8 steps (effective batch size = 8) |
| **Sequence Length** | 2048 tokens |
| **Output Directory** | `./fangyuan_qwen3_8b_lora` |

---

### 3. Interactive Terminal Chat (`test_model.py`)

Once training is complete, interact with Fang Yuan directly from your command line:

```bash
python test_model.py
```

#### Sample Interaction:
```text
==========================================================
             FANG YUAN EVALUATION CHAT             
==========================================================
Loading Base Model (Qwen/Qwen3-8B) + Trained LoRA Adapter...
Model successfully loaded! Type your question or dilemma (or 'exit' to quit).

You: I worked hard for years at my company, but someone else got the promotion through connections. Should I be angry?

Fang Yuan: Anger is merely the outward expression of one's own incompetence and impotence. 
You blame connections and unfairness, yet rules and connections are simply the mechanisms 
by which organizations distribute benefits. The righteous path speaks of fairness, but it 
is merely a banner to organize and control the majority. 

Reflect on your own value: if you are merely a convenient cog, you will be replaced when 
convenient. If this environment does not yield benefits, alter your methods or seek another 
domain. Regret and indignance will not grant you a single primeval stone.
--------------------------------------------------
```

---

### 4. Publishing to Hugging Face Hub (`push_to_hub.py`)

To share your trained adapter and automated model card with the open-source community:

1. Configure your Hugging Face token in `.env` (or run `huggingface-cli login`):
   ```bash
   cp .env.example .env
   # Edit .env and set HF_TOKEN=hf_...
   ```
2. Upload the model:
   ```bash
   python push_to_hub.py --repo_id <your-hf-username>/FangYuan-8B-LoRA
   ```

---

## 📊 Dataset Pipeline Architecture

For transparency and reproducibility, the dataset was synthesized through a rigorous multi-stage pipeline:

1. **Chapter Ingestion (`data/extract.py`)**: Extracted all chapters from digital sources into raw text.
2. **Text Normalization (`data/cleaner.py`)**: Stripped translator notes, footnotes, chapter numbers, ads, and corrupted text.
3. **Persona-Weighted Chunk Filtering (`generate_sft.py`)**: Scored novel chunks based on philosophical keyword density (*Ren Zu, benefits, eternal life, perseverance, demonic path*).
4. **Structured Generation**: Generated 10 distinct scenario-based pairs per chunk across the 5 persona archetypes using LLM multi-threading.
5. **Quality Filtering**: Automated length checks, role verification, and ChatML formatting.

---

## ⚖️ License & Ethical Notice

- **Code & Adapter Weights**: Licensed under the [Apache 2.0 License](LICENSE).
- **Novel Inspiration**: *Reverend Insanity* (蛊真人) is the intellectual work of author **Gu Zhen Ren**. This is an open-source, non-commercial fan research project exploring narrative alignment and philosophical personas in large language models.

