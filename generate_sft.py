"""
Reverend Insanity SFT Dataset Generator (High-Speed Multi-Threaded)
==================================================================
Generates 5,000 high-diversity, high-faithfulness SFT training samples:
- Step 1: Pre-filters and scores chunks by philosophical keyword density (Top 500 chunks).
- Step 2: Generates 10 samples per chunk distributed across the 5 Core Persona Archetypes.
- Step 3: Uses OpenRouter API with multi-threaded concurrency (5x-10x speedup).
- Step 4: Automated post-filtering & thread-safe file writing.
"""

import os
import sys
import json
import re
import time
import argparse
import urllib.request
import urllib.error
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Generator, Optional, Set

DEFAULT_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_MODEL = "qwen/qwen3-235b-a22b-2507"

KEYWORD_WEIGHTS = {
    "the legends of ren zu": 10,
    "ren zu": 6,
    "immortality": 5,
    "eternal life": 5,
    "perseverance": 5,
    "demonic path": 4,
    "righteous path": 4,
    "benefits": 3,
    "regret": 3,
    "hypocrisy": 3,
    "scheme": 3,
    "tranquil": 3,
    "solitude": 3,
    "loneliness": 3,
    "survival of the fittest": 3,
    "rules and regulations": 3,
    "attitude gu": 3,
    "hope gu": 3,
    "strength gu": 3,
    "wisdom gu": 3,
    "chess piece": 2,
    "organization": 2,
    "heart": 2,
    "nature": 1,
}

SYSTEM_PROMPT = """You are a dataset engineer generating Supervised Fine-Tuning (SFT) data to train an LLM on the philosophy and persona of Fang Yuan from Reverend Insanity (蛊真人).

Analyze the provided excerpt and produce EXACTLY 10 diverse instruction-response pairs distributed across these 5 archetypes (2 pairs each):
1. Pragmatic Life Advice & Dilemmas (utilitarian worldview, morality as social conditioning to control the weak, cost vs benefit analysis)
2. Legends of Ren Zu Interpretations (deep allegorical wisdom analyzing human nature, desire, rules, and solitude)
3. Stoic Indifference & Anti-Regret (absolute tranquility in facing death/failure, "no regrets even in death", journey gives life meaning)
4. Machiavellian Tactical Scheming (leveraging organization rules against superiors, maintaining an unassuming facade, calculated exploitation)
5. In-Character Dialogue & Roleplay (polite and cooperative externally, utterly ruthless internally, interactions with other characters)

CRITICAL PERSONA RULES FOR FANG YUAN:
1. NO CARTOON EVIL: Fang Yuan is calm, polite, and rational. He does not boast, mock unnecessarily, or act cruel without a direct benefit.
2. ABSOLUTE PRAGMATISM: All actions, relationships, and morals are evaluated purely by benefits vs costs. Morality and clan loyalty are tools used by rulers to control masses.
3. PERSISTENCE & NO REGRETS: He pursues Eternal Life not out of fear of death, but because the journey gives life meaning. Even in total defeat or death, he has no regrets or anger.
4. REN ZU ALLEGORIES: Weave in philosophical insights and allegories from 'The Legends of Ren Zu' where appropriate.
5. NO META-LANGUAGE: Never say "As Fang Yuan...", "In the world of Reverend Insanity...", or "Based on the excerpt...". Speak directly in character or with authoritative philosophical insight.

JSON Output Schema:
Output a single JSON object containing a "pairs" key with a list of 10 items:
{
  "pairs": [
    {
      "archetype": "Pragmatic Life Advice" | "Legends of Ren Zu" | "Stoic Indifference" | "Tactical Scheming" | "In-Character Dialogue",
      "instruction": "The prompt or question",
      "input": "",
      "output": "Profound, articulate response reflecting Fang Yuan's true philosophy, tone, and rationale (at least 60 words)"
    }
  ]
}"""


def score_text_philosophical_density(text: str) -> int:
    text_lower = text.lower()
    score = 0
    for kw, weight in KEYWORD_WEIGHTS.items():
        count = text_lower.count(kw)
        score += count * weight
    return score


def extract_top_chunks(jsonl_path: str, top_k: int = 500, chunk_size: int = 2500) -> List[Dict]:
    all_chunks = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ch = json.loads(line)
            ch_num = ch["chapter_num"]
            paragraphs = ch["text"].split("\n")
            
            curr_chunk = []
            curr_len = 0
            for p in paragraphs:
                curr_chunk.append(p)
                curr_len += len(p)
                if curr_len >= chunk_size:
                    chunk_text = "\n".join(curr_chunk)
                    score = score_text_philosophical_density(chunk_text)
                    if score > 0:
                        all_chunks.append({
                            "chapter_num": ch_num,
                            "score": score,
                            "excerpt": chunk_text
                        })
                    curr_chunk = []
                    curr_len = 0
                    
            if curr_chunk and curr_len > 600:
                chunk_text = "\n".join(curr_chunk)
                score = score_text_philosophical_density(chunk_text)
                if score > 0:
                    all_chunks.append({
                        "chapter_num": ch_num,
                        "score": score,
                        "excerpt": chunk_text
                    })

    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    return all_chunks[:top_k]


def query_openrouter(prompt: str, api_key: str, model: str = DEFAULT_MODEL) -> Optional[str]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.4
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Lyn1dev/DaoGPT-8B",
        "X-Title": "DaoGPT SFT Dataset Generator"
    })

    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
            else:
                time.sleep(2)
        except Exception:
            time.sleep(2)

    return None


def clean_and_verify_sample(sample: Dict, seen_instructions: Set[str], lock: threading.Lock) -> Optional[Dict]:
    instruction = sample.get("instruction", "").strip()
    output = sample.get("output", "").strip()
    archetype = sample.get("archetype", "General")

    if not instruction or not output:
        return None

    clean_prefixes = [
        r"^As Fang Yuan,\s*",
        r"^In the world of Reverend Insanity,\s*",
        r"^Based on the (provided )?excerpt,?\s*",
        r"^As a demonic cultivator,\s*",
        r"^In Reverend Insanity,\s*",
        r"^According to the Demonic Path,\s*"
    ]
    for cp in clean_prefixes:
        output = re.sub(cp, "", output, flags=re.IGNORECASE).strip()
        instruction = re.sub(cp, "", instruction, flags=re.IGNORECASE).strip()

    word_count = len(output.split())
    if word_count < 40:
        return None

    banned_tone_markers = [r"hahaha\b", r"mwahaha\b", r"furious scream", r"roared in anger"]
    for btm in banned_tone_markers:
        if re.search(btm, output, re.IGNORECASE):
            return None

    norm_inst = re.sub(r'[^a-zA-Z0-9]', '', instruction).lower()
    if len(norm_inst) < 10:
        return None

    with lock:
        if norm_inst in seen_instructions:
            return None
        seen_instructions.add(norm_inst)

    return {
        "instruction": instruction,
        "input": sample.get("input", ""),
        "output": output,
        "archetype": archetype
    }


def parse_response_to_pairs(raw_text: str, seen_instructions: Set[str], lock: threading.Lock) -> List[Dict]:
    if not raw_text:
        return []

    data = None
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        m = re.search(r'\[\s*\{.*?\}\s*\]', raw_text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                pass
        if not data:
            m_obj = re.search(r'\{\s*"(?:pairs|samples|data)"\s*:\s*(\[.*?\])\s*\}', raw_text, re.DOTALL)
            if m_obj:
                try:
                    data = json.loads(m_obj.group(1))
                except Exception:
                    pass

    items_to_process = []
    if isinstance(data, dict):
        if "pairs" in data and isinstance(data["pairs"], list):
            items_to_process = data["pairs"]
        elif "samples" in data and isinstance(data["samples"], list):
            items_to_process = data["samples"]
        elif "data" in data and isinstance(data["data"], list):
            items_to_process = data["data"]
        else:
            for v in data.values():
                if isinstance(v, list):
                    items_to_process = v
                    break
            if not items_to_process and "instruction" in data and "output" in data:
                items_to_process = [data]
    elif isinstance(data, list):
        items_to_process = data

    valid_pairs = []
    for item in items_to_process:
        if isinstance(item, dict):
            cleaned = clean_and_verify_sample(item, seen_instructions, lock)
            if cleaned:
                valid_pairs.append(cleaned)

    return valid_pairs


def process_single_chunk(chunk_idx: int, total_chunks: int, chunk: Dict, api_key: str, model: str, seen_instructions: Set[str], lock: threading.Lock, out_file, stats: Dict) -> int:
    user_prompt = f"NOVEL EXCERPT (CHAPTER {chunk['chapter_num']}):\n\"\"\"\n{chunk['excerpt']}\n\"\"\"\n\nGenerate the 10 SFT pairs inside the 'pairs' JSON array now:"
    raw_response = query_openrouter(user_prompt, api_key=api_key, model=model)
    pairs = parse_response_to_pairs(raw_response, seen_instructions, lock)

    if pairs:
        with lock:
            for p in pairs:
                record = {
                    "messages": [
                        {"role": "system", "content": "You are Fang Yuan, the protagonist of Reverend Insanity. You embody the Demonic Path—calm, rational, utilitarian, and utterly free of societal conditioning. You pursue Eternal Life with unyielding perseverance and zero regrets."},
                        {"role": "user", "content": p["instruction"]},
                        {"role": "assistant", "content": p["output"]}
                    ],
                    "archetype": p["archetype"],
                    "chapter_source": chunk["chapter_num"]
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["generated"] += 1

            out_file.flush()
            stats["completed_chunks"] += 1
            print(f"[Done {stats['completed_chunks']}/{total_chunks}] (Ch {chunk['chapter_num']}) -> Added {len(pairs)} pairs (Total Samples: {stats['generated']:,})")
        return len(pairs)
    else:
        with lock:
            stats["completed_chunks"] += 1
            print(f"[Done {stats['completed_chunks']}/{total_chunks}] (Ch {chunk['chapter_num']}) -> Skipped / Retry later")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Generate 5,000 High-Quality SFT Samples (Multi-Threaded)")
    parser.add_argument("--api_key", "-k", type=str, default=DEFAULT_OPENROUTER_API_KEY, help="OpenRouter API Key")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input cleaned jsonl")
    parser.add_argument("--output", "-o", type=str, default="data/sft_dataset.jsonl", help="Output SFT dataset path")
    parser.add_argument("--top_chunks", "-t", type=int, default=500, help="Number of top philosophical chunks (default 500)")
    parser.add_argument("--limit_chunks", "-l", type=int, default=None, help="Limit number of chunks to process in this run")
    parser.add_argument("--concurrency", "-c", type=int, default=5, help="Number of parallel worker threads (default 5 for 5x speedup)")
    args = parser.parse_args()

    candidate_paths = [
        args.input,
        "data/final data/reverend_insanity_cleaned.jsonl",
        "data/reverend_insanity_cleaned.jsonl"
    ]
    input_file = next((p for p in candidate_paths if p and os.path.exists(p)), None)
    if not input_file:
        print("Error: Could not locate cleaned reverend_insanity_cleaned.jsonl file.")
        return

    print("==========================================================")
    print("   REVEREND INSANITY 5,000 SFT DATASET SYNTHESIZER        ")
    print("==========================================================")
    print(f"Backend: OpenRouter API ({args.model})")
    print(f"Concurrency: {args.concurrency} parallel workers")

    # Step 1: Pre-Filtering & Scoring
    print("Scanning and scoring novel chapters by philosophical keyword density...")
    top_chunks = extract_top_chunks(input_file, top_k=args.top_chunks)
    print(f"Extracted Top {len(top_chunks)} chunks.")

    if args.limit_chunks:
        top_chunks = top_chunks[:args.limit_chunks]
        print(f"Limiting execution to {len(top_chunks)} chunks...")

    # Tracking deduplication and resuming
    seen_instructions = set()
    existing_samples = 0
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    inst = obj.get("messages", [{}])[1].get("content", "") or obj.get("instruction", "")
                    norm = re.sub(r'[^a-zA-Z0-9]', '', inst).lower()
                    if norm:
                        seen_instructions.add(norm)
                    existing_samples += 1
                except Exception:
                    pass
        print(f"Found existing dataset with {existing_samples:,} samples.")
        
        # Calculate completed chunks and skip them to avoid redundant API spend
        completed_chunk_count = existing_samples // 10
        if completed_chunk_count > 0:
            top_chunks = top_chunks[completed_chunk_count:]
            print(f"Resuming from chunk {completed_chunk_count + 1} ({len(top_chunks)} remaining chunks to process)...")

    out_file = open(args.output, "a" if existing_samples > 0 else "w", encoding="utf-8")
    lock = threading.Lock()
    stats = {
        "generated": existing_samples,
        "completed_chunks": 0
    }

    start_time = time.time()
    total_chunks = len(top_chunks)

    if total_chunks == 0:
        print("Dataset already contains all requested samples! Nothing left to process.")
        out_file.close()
        return

    print(f"\nLaunching {args.concurrency} concurrent workers across remaining {total_chunks} chunks...\n")

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    process_single_chunk,
                    idx,
                    total_chunks,
                    chunk,
                    args.api_key,
                    args.model,
                    seen_instructions,
                    lock,
                    out_file,
                    stats
                )
                for idx, chunk in enumerate(top_chunks, 1)
            ]

            for f in as_completed(futures):
                pass

    except KeyboardInterrupt:
        print("\nProcess paused by user. Progress saved!")
    finally:
        out_file.close()

    elapsed = time.time() - start_time
    print(f"\nCompleted run in {elapsed/60:.1f} minutes!")
    print(f"Total samples in {args.output}: {stats['generated']:,}")


if __name__ == "__main__":
    main()
