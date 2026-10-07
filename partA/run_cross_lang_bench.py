#!/usr/bin/env python3
"""
========================================================================================
ASSIGNMENT PART A — Corrected Multilingual Cross-Language Benchmark
========================================================================================
QUESTION:
"Re-run the comparison using at least two tokenizers (one of them multilingual/Indic-aware)
 and at least two denominators (e.g. per whitespace word, per grapheme cluster, per UTF-8 byte,
 per parallel sentence).
 Then answer the question that actually matters:
 which single number should drive a routing-and-cost decision, and why?
 (Hint: think hard about what the denominator is supposed to hold constant across languages)."

CORE QUESTION:
"Does our tokenizer really behave worse/better for different languages, and are we
measuring that fairly?"
----------------------------------------------------------------------------------------
SOLUTION IMPLEMENTED IN THIS SCRIPT:
1. Tokenizers Evaluated:
   - gpt2: 50,257 vocab (tiktoken, English-only baseline used by previous intern)
   - xlm-roberta-base: 250,002 vocab (Hugging Face, multilingual standard)
   - Qwen/Qwen2.5-0.5B: 151,665 vocab (Hugging Face, modern Byte-level BPE LLM)

2. Denominators Evaluated (4 Denominators):
   - Per Sentence (Semantic unit): Ground truth semantic task holding meaning constant.
   - Per Whitespace Word: Highlights agglutinative distortion in Dravidian languages.
   - Per Character: Highlights orthographic character compression.
   - Per UTF-8 Byte: Measures raw byte-level information density.

3. Core Finding:
   - Under gpt2: Hindi is 7.42x English, Kannada 13.58x, Tamil 15.54x (byte fallback).
   - Under xlm-roberta-base: Hindi is only 1.25x English, Kannada 1.35x, Tamil 1.35x!
   - Best Metric: TOKENS PER PARALLEL SENTENCE (SEMANTIC TASK), because user utility
     and prompt/completion workloads scale with meaning, not arbitrary word boundaries.
========================================================================================
"""

import os
import sys
import csv
import unicodedata
import tiktoken
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
LANGS = ["eng", "hin", "kan", "tam", "tel"]

def load_clean_corpus():
    data = {}
    for lang in LANGS:
        path = os.path.join(CORPUS_DIR, f"{lang}.txt")
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(unicodedata.normalize("NFC", line))
        data[lang] = lines
    return data


def get_tokenizers():
    tokenizers = {}
    
    # 1. GPT-2 (English-only baseline)
    enc_gpt2 = tiktoken.get_encoding("gpt2")
    tokenizers["gpt2"] = lambda s: enc_gpt2.encode(s)
    
    # 2. XLM-RoBERTa (Multilingual standard)
    tok_xlm = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenizers["xlm-roberta-base"] = lambda s: tok_xlm.encode(s, add_special_tokens=False)
    
    # 3. Qwen 2.5 (Modern LLM tokenizer)
    tok_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizers["Qwen/Qwen2.5-0.5B"] = lambda s: tok_qwen.encode(s, add_special_tokens=False)
    
    return tokenizers


def main():
    corpus = load_clean_corpus()
    num_sentences = len(corpus["eng"])
    print(f"Loaded {num_sentences} parallel sentences for {list(corpus.keys())}\n")
    
    tokenizers = get_tokenizers()
    
    all_results = []
    
    for tok_name, encode_fn in tokenizers.items():
        print("="*80)
        print(f"TOKENIZER: {tok_name}")
        print("="*80)
        print(f"{'Lang':<6}{'Total Toks':<12}{'Tok/Sent':<12}{'Tok/Word':<12}{'Tok/Char':<12}{'Tok/Byte':<12}{'Rel to Eng':<12}")
        print("-" * 80)
        
        eng_tokens = None
        eng_tok_per_word = None
        
        for lang in LANGS:
            lines = corpus[lang]
            total_toks = 0
            total_words = 0
            total_chars = 0
            total_bytes = 0
            
            for line in lines:
                toks = encode_fn(line)
                words = line.split()
                total_toks += len(toks)
                total_words += len(words)
                total_chars += len(line)
                total_bytes += len(line.encode("utf-8"))
                
            tok_per_sent = total_toks / len(lines)
            tok_per_word = total_toks / total_words
            tok_per_char = total_toks / total_chars
            tok_per_byte = total_toks / total_bytes
            
            if lang == "eng":
                eng_tokens = total_toks
                eng_tok_per_word = tok_per_word
                rel_eng = 1.0
                rel_word = 1.0
            else:
                rel_eng = total_toks / eng_tokens
                rel_word = tok_per_word / eng_tok_per_word
                
            all_results.append({
                "tokenizer": tok_name,
                "lang": lang,
                "total_tokens": total_toks,
                "num_sentences": len(lines),
                "total_words": total_words,
                "total_chars": total_chars,
                "total_bytes": total_bytes,
                "tok_per_sent": round(tok_per_sent, 2),
                "tok_per_word": round(tok_per_word, 3),
                "tok_per_char": round(tok_per_char, 3),
                "tok_per_byte": round(tok_per_byte, 3),
                "ratio_vs_eng_semantic": round(rel_eng, 2),
                "ratio_vs_eng_word": round(rel_word, 2)
            })
            
            print(f"{lang:<6}{total_toks:<12}{tok_per_sent:<12.2f}{tok_per_word:<12.2f}{tok_per_char:<12.3f}{tok_per_byte:<12.3f}{rel_eng:<12.2f}x")
        print()

    # Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "results", "cross_language_benchmark.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Benchmark results saved to {csv_path}")


if __name__ == "__main__":
    main()
