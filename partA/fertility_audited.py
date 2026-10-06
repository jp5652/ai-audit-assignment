#!/usr/bin/env python3
"""
fertility_audited.py -- Audited and corrected tokenizer fertility benchmark.

Fixes all identified bugs in fertility.py:
  1. Fixes whitespace splitting: Uses line.split() instead of line.split(" ")
     to avoid empty strings on consecutive spaces.
  2. Preserves casing: Removed line.lower() to avoid distorting case-sensitive
     BPE tokenizers (e.g. acronyms and proper nouns).
  3. Micro-averaging (Ratio of Totals): Reports aggregate micro-averages
     (total tokens / total units) for accurate serving capacity estimation,
     alongside macro-averages for per-line variance analysis.
  4. Preserves NFC normalization: Ensures canonical Unicode representation to
     prevent token explosion on decomposed text.
  5. Correct special token handling: add_special_tokens=False to measure text
     representation without constant-offset sequence biases.
  6. Multi-denominator support: Computes tokens per parallel sentence
     (semantic unit), per word, per character, and per UTF-8 byte.

Usage:
  python fertility_audited.py --corpus eng=corpus/eng.txt \
                             --corpus hin=corpus/hin.txt \
                             --corpus kan=corpus/kan.txt \
                             --tokenizer hf:xlm-roberta-base
"""

import argparse
import os
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")


def load_tokenizer(spec: str):
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    else:
        import tiktoken
        enc = tiktoken.get_encoding(spec)
        return enc.encode


def read_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Canonical Unicode NFC normalization is essential for Indic text
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


def analyze(lines, encode):
    """
    Computes both micro-averaged (total/total) and macro-averaged metrics
    across sentences, words, characters, and UTF-8 bytes.
    """
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_bytes = 0

    per_line_fert_word = []
    per_line_fert_char = []

    for line in lines:
        tokens = encode(line)
        # BUG FIX 1: line.split() strips leading/trailing/multiple whitespaces cleanly
        # BUG FIX 2: Preserved case -- do not call line.lower()
        words = line.split()
        n_tok = len(tokens)
        n_word = len(words)
        n_char = len(line)
        n_byte = len(line.encode("utf-8"))

        total_tokens += n_tok
        total_words += n_word
        total_chars += n_char
        total_bytes += n_byte

        per_line_fert_word.append(n_tok / n_word if n_word > 0 else 0)
        per_line_fert_char.append(n_tok / n_char if n_char > 0 else 0)

    n_lines = len(lines)
    micro = {
        "num_lines": n_lines,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "tok_per_sentence": total_tokens / n_lines,
        "tok_per_word": total_tokens / total_words,
        "tok_per_char": total_tokens / total_chars,
        "tok_per_byte": total_tokens / total_bytes,
    }
    macro = {
        "macro_fert_word": sum(per_line_fert_word) / n_lines,
        "macro_fert_char": sum(per_line_fert_char) / n_lines,
    }
    return micro, macro


def main():
    ap = argparse.ArgumentParser(description="Audited Tokenizer Benchmark")
    ap.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LANG=PATH",
        help="language code and file path, e.g. eng=corpus/eng.txt",
    )
    ap.add_argument("--tokenizer", default="gpt2", help="tiktoken name or hf:<repo_id>")
    args = ap.parse_args()

    encode = load_tokenizer(args.tokenizer)

    print(f"Tokenizer: {args.tokenizer}")
    print(
        f"{'Lang':<6}{'Sentences':<10}{'Total Toks':<12}"
        f"{'Tok/Sent':<12}{'Tok/Word':<12}{'Tok/Char':<12}{'Tok/Byte':<12}"
    )
    print("-" * 76)

    results = {}
    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        lines = read_lines(path)
        micro, macro = analyze(lines, encode)
        results[lang] = micro
        print(
            f"{lang:<6}{micro['num_lines']:<10}{micro['total_tokens']:<12}"
            f"{micro['tok_per_sentence']:<12.2f}{micro['tok_per_word']:<12.2f}"
            f"{micro['tok_per_char']:<12.3f}{micro['tok_per_byte']:<12.3f}"
        )

    if len(results) >= 2:
        langs = list(results)
        base = langs[0]
        base_tok_sent = results[base]["tok_per_sentence"]
        base_tok_word = results[base]["tok_per_word"]
        print("\nRelative Comparison vs Base (" + base + "):")
        print(f"{'Lang':<6}{'Semantic Ratio (Tok/Sent)':<30}{'Word Ratio (Tok/Word)':<25}")
        print("-" * 65)
        for lang in langs[1:]:
            sem_ratio = results[lang]["tok_per_sentence"] / base_tok_sent
            word_ratio = results[lang]["tok_per_word"] / base_tok_word
            print(
                f"{lang:<6}{sem_ratio:<30.2f}x{word_ratio:<25.2f}x"
            )


if __name__ == "__main__":
    main()
