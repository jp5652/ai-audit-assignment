#!/usr/bin/env python3
"""
run_cross_lang_bench.py -- Corrected multilingual cross-language tokenizer benchmark.

Evaluates 5 parallel languages (eng, hin, kan, tam, tel) across 1,012 FLORES-200 sentences.
Tokenizers:
  1. gpt2 (tiktoken, 50k vocab, English-only baseline)
  2. xlm-roberta-base (SentencePiece, 250k vocab, multilingual)
  3. Qwen/Qwen2.5-0.5B (Byte-level BPE, 152k vocab, modern LLM tokenizer)

Denominators evaluated:
  - per sentence (Semantic unit: holds meaning constant)
  - per whitespace word (Morphological unit: highlights agglutinative distortion)
  - per character (Orthographic unit)
  - per UTF-8 byte (Information-theoretic compression unit)
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
    
    # 1. GPT-2
    enc_gpt2 = tiktoken.get_encoding("gpt2")
    tokenizers["gpt2"] = lambda s: enc_gpt2.encode(s)
    
    # 2. XLM-RoBERTa
    tok_xlm = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenizers["xlm-roberta-base"] = lambda s: tok_xlm.encode(s, add_special_tokens=False)
    
    # 3. Qwen 2.5
    tok_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizers["Qwen/Qwen2.5-0.5B"] = lambda s: tok_qwen.encode(s, add_special_tokens=False)
    
    return tokenizers


def main():
    corpus = load_clean_corpus()
    num_sentences = len(corpus["eng"])
    print(f"Loaded {num_sentences} parallel sentences for {list(corpus.keys())}\n")
    
    tokenizers = get_tokenizers()
    
    # Storage for summary rows
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
