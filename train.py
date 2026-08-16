"""
FangYuan-8B Fine-Tuning Script (Qwen3-8B 1-Epoch Full Run)
==========================================================
- Base Model: Qwen/Qwen3-8B
- Dataset: All 4,901 samples from data/sft_dataset.jsonl
- Schedule: 1 Full Epoch (~612 optimization steps in 1 continuous run)
- Hardware: Optimized for RTX 4060 Ti 8GB VRAM (4-bit NF4 + Paged 8-bit AdamW)
"""

import os
import sys
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# 1. Hardware Verification
print("==========================================================")
print("     FANGYUAN-8B QWEN3-8B TRAINER (1 FULL EPOCH)          ")
print("==========================================================")
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. Please run with `py -3.10 train.py`.")
    sys.exit(1)

device_name = torch.cuda.get_device_name(0)
print(f"Detected GPU: {device_name} (CUDA {torch.version.cuda})")

# 2. Model & Training Settings
model_id = "Qwen/Qwen3-8B"
max_seq_length = 2048
output_dir = "./fangyuan_qwen3_8b_lora"

print(f"Base Model: {model_id}")
print(f"Quantization: 4-bit NF4 (8GB VRAM Optimized)")

# 3. 4-Bit BitsAndBytes Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 4. Load Tokenizer & Model
print("Loading Tokenizer and Qwen3-8B weights...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# 5. QLoRA Adapter (All Linear Attention & MLP Layers)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, peft_config)
print("\nTrainable Parameters:")
model.print_trainable_parameters()

# 6. Load Dataset (All 4,901 Samples)
script_dir = os.path.dirname(os.path.abspath(__file__))
candidate_paths = [
    os.path.join(script_dir, "data", "sft_dataset.jsonl"),
    os.path.join(script_dir, "sft_dataset.jsonl"),
    os.path.join(script_dir, "..", "data", "sft_dataset.jsonl"),
    "data/sft_dataset.jsonl",
    "sft_dataset.jsonl"
]
data_file = next((p for p in candidate_paths if os.path.exists(p)), "data/sft_dataset.jsonl")
print(f"\nLoading dataset from: {data_file}")

with open(data_file, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f if line.strip()]

total_samples = len(raw_data)
effective_batch_size = 8
steps_per_epoch = total_samples // effective_batch_size
print(f"Total training samples: {total_samples:,}")
print(f"Training Schedule: 1 Full Epoch = {steps_per_epoch} optimization steps")

# Format into ChatML prompt strings using tokenizer chat template
formatted_texts = []
for item in raw_data:
    messages = item.get("messages", [])
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    formatted_texts.append({"text": text})

train_dataset = Dataset.from_list(formatted_texts)

# 7. Training Arguments (1 Full Epoch, Optimized for 8GB VRAM)
training_args = SFTConfig(
    output_dir=output_dir,
    max_length=max_seq_length,
    dataset_text_field="text",
    per_device_train_batch_size=1,        # 1 sample per batch to stay within 8GB VRAM
    gradient_accumulation_steps=8,         # Effective batch size = 8
    num_train_epochs=1,                   # 1 Full Epoch across all samples
    learning_rate=2e-4,
    warmup_steps=18,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=5,
    optim="paged_adamw_8bit",              # Paged 8-bit AdamW
    gradient_checkpointing=True,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    seed=3407,
    report_to="none",
)

# 8. Start SFT Training
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
)

print(f"\nStarting 1-Epoch Training Run ({steps_per_epoch} total steps)...\n")
trainer.train()

# 9. Save Trained LoRA Adapter
print(f"\nTraining Complete! Saving adapter weights to: {output_dir}")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Done! Adapter successfully saved to {output_dir}.")
