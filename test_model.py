"""
Interactive Chat & Evaluation for DaoGPT (Fang Yuan Qwen3-8B)
============================================================
Loads the base Qwen3-8B model + your trained LoRA adapter and lets
you chat with Fang Yuan directly in your terminal.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = "./fangyuan_qwen3_8b_lora"

print("==========================================================")
print("             DAOGPT FANG YUAN EVALUATION CHAT             ")
print("==========================================================")
print(f"Loading Base Model ({BASE_MODEL}) + Trained LoRA Adapter...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("\nModel successfully loaded! Type your question or dilemma (or 'exit' to quit).\n")

system_prompt = "You are Fang Yuan, the protagonist of Reverend Insanity. You embody the Demonic Path—calm, rational, utilitarian, and utterly free of societal conditioning. You pursue Eternal Life with unyielding perseverance and zero regrets."

history = [{"role": "system", "content": system_prompt}]

while True:
    try:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        history.append({"role": "user", "content": user_input})

        # Apply ChatML template
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Slice generated tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        print(f"\nFang Yuan: {response}\n" + "-" * 50)
        history.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("\nSession ended.")
        break
