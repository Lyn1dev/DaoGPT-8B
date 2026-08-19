"""
Interactive Chat & Evaluation for FangYuan-8B (Fang Yuan Qwen3-8B)
==================================================================
Loads base Qwen3-8B model + trained LoRA adapter and launches an
interactive terminal chat with Fang Yuan.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = "./fangyuan_qwen3_8b_lora"

print("==========================================================")
print("             FANG YUAN EVALUATION CHAT                    ")
print("==========================================================")
print(f"Loading Base Model ({BASE_MODEL}) + Trained LoRA Adapter...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("\n" + "=" * 58)
print(" FangYuan-8B loaded successfully! ")
print(" You can now chat directly with Fang Yuan in your terminal.")
print(" Type 'exit' or 'quit' at any time to leave.")
print("=" * 58 + "\n")

# Default Persona Prompt
system_prompt = "You are Fang Yuan, the protagonist of Reverend Insanity. You embody the Demonic Path—calm, rational, utilitarian, and utterly free of societal conditioning. You pursue Eternal Life with unyielding perseverance and zero regrets."

history = [{"role": "system", "content": system_prompt}]

while True:
    try:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nLeaving conversation...")
            break

        history.append({"role": "user", "content": user_input})

        # Format input through ChatML template
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=450,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        print(f"\nFang Yuan:\n{response}\n" + "-" * 58)
        history.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("\nSession ended.")
        break
