

import os
import sys
import re
import html
import zipfile
import argparse
from typing import List, Tuple, Optional

# Regex patterns for matching unwanted metadata and commentary
RE_CHAPTER_HEADER = re.compile(
    r'^\s*(?:Chapter|Volume|Book)\s+\d+', 
    re.IGNORECASE
)

RE_CHAPTER_NUM_TITLE = re.compile(
    r'^\s*\d{1,4}\s+[A-Z]',
)

RE_CREDIT_LINE = re.compile(
    r'^\s*(?:Translator|Editor|Translated\s+by|Edited\s+by|Proofreader|TL|ED)\s*:',
    re.IGNORECASE
)

RE_NOTE_START = re.compile(
    r'^\s*(\[|\()?\s*(?:tl\s*note|ed\s*note|translator[\'’]?s?\s*note|editor[\'’]?s?\s*note|author[\'’]?s?\s*note|translator[\'’]?s?\s*thoughts|editor[\'’]?s?\s*thoughts|t\/n|e\/n|a\/n|tn\s*note|t\/l\s*note|footnotes?)\b',
    re.IGNORECASE
)

RE_TRANSLATOR_COMMENT = re.compile(
    r'^\s*(?:ChibiGeneral|Skyfarrow|Atlas\s*Studios|Sigma|–\s*Skyfarrow|\*\s*This\s*chapter\s*was\s*brought\s*to\s*you\s*by|Credits\s*to\s*Chibigen|\*\s*Donations\s*are)\b',
    re.IGNORECASE
)

RE_FOOTNOTE_LINE = re.compile(
    r'^\s*(?:\(\d{1,2}\)[\.:]?|\[\d{1,2}\][\.:]?|\d{1,2}[\.:])\s+',
    re.IGNORECASE
)

RE_PROMO_URL = re.compile(
    r'(patreon\.com|discord\.gg|paypal\.me|ko-fi\.com|qidian\.com|webnovel\.com|9kafe\.com|boxnovel\.com|novelfull\.com|novelupdates\.com)',
    re.IGNORECASE
)

KNOWN_CREDIT_NAMES = {'skyfarrow', 'chibigeneral', 'chibigen', 'atlas studios', 'sigma'}


def clean_paragraph_text(p: str) -> str:

    if not p:
        return ""
    
    # 1. Unescape HTML entities (&quot;, &lt;, &gt;, &nbsp;, etc.)
    p = html.unescape(p)
    
    # 2. Strip any leftover HTML tags
    p = re.sub(r'<[^>]+>', '', p)
    
    # 3. Normalize non-standard spaces
    p = p.replace('\xa0', ' ').replace('\u3000', ' ').replace('\ufeff', '').replace('\u200b', '')
    
    # 4. Strip leading translator/editor credit prefix if glued to start of sentence
    # e.g., "Translator:ChibiGeneral | Editor: ChibiGeneral Story text here..."
    credit_match = re.match(r'^\s*(?:(?:Translator|Editor)\s*:[^|:\n“"\'\(\[]*)+[|:\s]*', p, re.IGNORECASE)
    if credit_match:
        matched_str = credit_match.group(0)
        if len(matched_str) < len(p):
            p = p[len(matched_str):].strip()
            
    # Remove leading ": Skyfarrow" or ": Atlas Studios" artifact
    p = re.sub(r'^\s*:\s*(?:Skyfarrow|Atlas\s*Studios|ChibiGeneral|Sigma)\s*', '', p, flags=re.IGNORECASE)
            
    # 5. Remove inline bracketed notes: [TL Note: ...], (TL Note: ...), [T/N: ...]
    p = re.sub(r'\[\s*(?:TL|ED|Translator|Editor|Author|T/N|E/N|A/N)[^\]]*\]', '', p, flags=re.IGNORECASE)
    p = re.sub(r'\(\s*(?:TL|ED|Translator|Editor|Author|T/N|E/N|A/N)[^\)]*\)', '', p, flags=re.IGNORECASE)
    
    # 6. Remove footnote citation markers like [1], (1), attached to words
    p = re.sub(r'(?<=\w)\[\d{1,2}\]', '', p)
    p = re.sub(r'(?<=\w)\(\d{1,2}\)', '', p)
    p = re.sub(r'\[\d{1,2}\]', '', p)
    
    # 7. Normalize multi-spaces
    p = re.sub(r'[ \t]+', ' ', p).strip()
    return p


def is_unwanted_paragraph(p: str) -> bool:

    p_strip = p.strip()
    if not p_strip:
        return True
    if p_strip.lower() in KNOWN_CREDIT_NAMES or p_strip.lower().lstrip(': ').strip() in KNOWN_CREDIT_NAMES:
        return True
    if RE_CHAPTER_HEADER.match(p_strip):
        return True
    if RE_CHAPTER_NUM_TITLE.match(p_strip):
        return True
    if RE_CREDIT_LINE.match(p_strip) and len(p_strip) < 150:
        return True
    if RE_NOTE_START.match(p_strip):
        return True
    if RE_TRANSLATOR_COMMENT.match(p_strip):
        return True
    if RE_FOOTNOTE_LINE.match(p_strip) and ('–' in p_strip or '-' in p_strip or ':' in p_strip or len(p_strip) < 250):
        return True
    if RE_PROMO_URL.search(p_strip) and len(p_strip) < 250:
        return True
    if p_strip.lower() in [
        'notes', 'note', 'footnotes', 'footnote', 'author thoughts', 
        'author note', 'translator thoughts', 'translator’s thoughts:', 
        'translator’s thoughts', "translator's thoughts"
    ]:
        return True
    if p_strip in ['—', '–', '-', '***', '---', '___', '...']:
        return True
    return False


def extract_chapter_from_epub(epub_path: str, page_name: str) -> List[str]:

    with zipfile.ZipFile(epub_path, 'r') as z:
        content = z.read(page_name).decode('utf-8', errors='ignore')
        
    # If <hr /> separates the body from translator footnotes at the end
    if '<hr' in content:
        parts = re.split(r'<hr\s*\/?>', content, flags=re.IGNORECASE)
        if len(parts) > 1 and any(k in parts[1].lower() for k in ['note', 'translator', 'chibigeneral', 'skyfarrow', 'gu yue:', '(1)', '[1]']):
            content = parts[0]
            
    raw_paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', content, flags=re.DOTALL)
    if not raw_paragraphs:
        body = re.search(r'<body[^>]*>(.*?)</body>', content, flags=re.DOTALL)
        if body:
            raw_paragraphs = body.group(1).split('\n')
            
    cleaned_paras = []
    for raw_p in raw_paragraphs:
        p = clean_paragraph_text(raw_p)
        if p:
            cleaned_paras.append(p)
            
    # Strip leading chapter headers, credits, etc.
    while cleaned_paras and is_unwanted_paragraph(cleaned_paras[0]):
        cleaned_paras.pop(0)
        
    # Strip trailing translator notes / comment blocks
    for idx in range(len(cleaned_paras) - 1, max(-1, len(cleaned_paras) - 10), -1):
        if RE_NOTE_START.match(cleaned_paras[idx]) or RE_TRANSLATOR_COMMENT.match(cleaned_paras[idx]):
            cleaned_paras = cleaned_paras[:idx]
            break
            
    # Strip trailing unwanted paragraphs
    while cleaned_paras and is_unwanted_paragraph(cleaned_paras[-1]):
        cleaned_paras.pop()
        
    # Filter out any mid-chapter note paragraphs
    final_paras = [p for p in cleaned_paras if not is_unwanted_paragraph(p)]
    return final_paras


def process_epubs(epub_files: List[str]) -> List[List[str]]:

    all_chapters = []
    total_epubs = len(epub_files)
    
    for i, epub_path in enumerate(epub_files, 1):
        print(f"[{i}/{total_epubs}] Extracting from {os.path.basename(epub_path)}...")
        with zipfile.ZipFile(epub_path, 'r') as z:
            pages = [f for f in z.namelist() if f.startswith('OEBPS/page-') and f.endswith('.html')]
            pages.sort(key=lambda x: int(re.search(r'page-(\d+)\.html', x).group(1)))
            
            for page in pages:
                paras = extract_chapter_from_epub(epub_path, page)
                if paras:
                    all_chapters.append(paras)
                    
    return all_chapters


def find_default_epubs(base_dir: str) -> List[str]:

    expected_order = [
        "reverend-insanity-c1-c500.epub",
        "reverend-insanity-c501-1000.epub",
        "reverend-insanity-c1001-1500.epub",
        "reverend-insanity-c1501-c2000.epub",
        "reverend-insanity-c2001-end.epub"
    ]
    
    search_dirs = [
        os.path.join(base_dir, "data"),
        base_dir,
        os.path.join(base_dir, "..", "data")
    ]
    
    for d in search_dirs:
        if os.path.exists(d):
            found = [os.path.join(d, fname) for fname in expected_order if os.path.exists(os.path.join(d, fname))]
            if len(found) == len(expected_order):
                return found
                
    return []


def main():
    parser = argparse.ArgumentParser(description="Clean Reverend Insanity dataset for AI training.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path for cleaned output file (.txt or .jsonl)")
    parser.add_argument("--format", "-f", type=str, choices=["txt", "jsonl", "both"], default="both", help="Output format (txt, jsonl, or both)")
    parser.add_argument("--epub_dir", "-e", type=str, default=None, help="Directory containing epub files")
    args = parser.parse_args()

    # Determine base directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..")) if os.path.basename(script_dir) == "data" else script_dir
    data_dir = os.path.join(project_root, "data")
    
    epub_search_dir = args.epub_dir if args.epub_dir else data_dir
    epub_files = find_default_epubs(epub_search_dir)
    
    if not epub_files:
        print(f"Error: Could not locate EPUB files in '{epub_search_dir}'.")
        print("Please ensure the 5 EPUB files are located in the data/ directory.")
        sys.exit(1)

    print(f"Found {len(epub_files)} volume EPUB files.")
    
    # Process EPUBs
    chapters = process_epubs(epub_files)
    total_chapters = len(chapters)
    total_paragraphs = sum(len(c) for c in chapters)
    total_words = sum(sum(len(p.split()) for p in c) for c in chapters)
    total_chars = sum(sum(len(p) for p in c) for c in chapters)

    print("\nExtraction & Cleaning Complete!")
    print(f"  • Total Chapters:   {total_chapters:,}")
    print(f"  • Total Paragraphs: {total_paragraphs:,}")
    print(f"  • Total Word Count: {total_words:,}")
    print(f"  • Total Characters: {total_chars:,}")
    print(f"  • Estimated Tokens: ~{int(total_words * 1.33):,} tokens")

    # Determine output file paths
    out_txt = os.path.join(data_dir, "reverend_insanity_cleaned.txt")
    out_jsonl = os.path.join(data_dir, "reverend_insanity_cleaned.jsonl")
    
    if args.output:
        if args.output.endswith(".jsonl"):
            out_jsonl = args.output
            args.format = "jsonl"
        else:
            out_txt = args.output

    # Write Plain Text
    if args.format in ["txt", "both"]:
        print(f"\nWriting plain text dataset to: {out_txt}")
        with open(out_txt, "w", encoding="utf-8") as f:
            for i, ch_paras in enumerate(chapters):
                chapter_text = "\n".join(ch_paras)
                f.write(chapter_text)
                if i < len(chapters) - 1:
                    f.write("\n\n")
        print(f"  -> Successfully written {os.path.getsize(out_txt):,} bytes to {os.path.basename(out_txt)}")

    # Write JSONL
    if args.format in ["jsonl", "both"]:
        import json
        print(f"\nWriting JSONL dataset to: {out_jsonl}")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for i, ch_paras in enumerate(chapters, 1):
                chapter_text = "\n".join(ch_paras)
                record = {
                    "chapter_num": i,
                    "text": chapter_text,
                    "paragraph_count": len(ch_paras),
                    "word_count": len(chapter_text.split())
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  -> Successfully written {os.path.getsize(out_jsonl):,} bytes to {os.path.basename(out_jsonl)}")

    print("\nDone! Dataset is clean and ready for AI training.")


if __name__ == "__main__":
    main()
