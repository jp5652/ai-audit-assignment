#!/usr/bin/env python3
"""
========================================================================================
ASSIGNMENT PART A — Audited & Corrected fertility.py Script
========================================================================================
QUESTION:
"Audit fertility.py and replace it with a corrected script.
 Recompute the analysis using at least 2 tokenizers and at least 2 meaningful denominators.
 Answer which single number should drive routing and cost decisions."

CORE QUESTION:
"Does our tokenizer really behave worse/better for different languages, and are we
measuring that fairly?"
----------------------------------------------------------------------------------------
BUGS FIXED IN THIS SCRIPT:
1. Whitespace Bug Fix: Uses line.split() instead of line.split(" ") to strip consecutive
   spaces without generating empty strings.
2. Case Distortion Fix: Removed line.lower() to preserve cased BPE tokens (e.g. acronyms).
3. Statistical Aggregation Fix: Reports aggregate Micro-averages (total tokens / total units)
   for serving capacity, alongside Macro-averages.
4. Preserved NFC Unicode Normalization: Essential to prevent token explosion on decomposed text.
5. Preserved add_special_tokens=False: Prevents sequence-length constant offset distortion.
6. Multi-Denominator Support: Computes Tokens/Sentence, Tokens/Word, Tokens/Char, Tokens/Byte.

Usage:
  python fertility_audited.py --corpus eng=corpus/eng.txt \
                             --corpus hin=corpus/hin.txt \
                             --corpus kan=corpus/kan.txt \
                             --tokenizer hf:xlm-roberta-base
========================================================================================
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
        # SUSPICIOUS BUT CORRECT: add_special_tokens=False avoids constant BOS/EOS skew
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
            # SUSPICIOUS BUT CORRECT: NFC normalization is standard Unicode (UAX #15)
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
    if n_lines == 0:
        raise ValueError("Corpus is empty: no valid non-empty lines found to analyze.")

    micro = {
        "num_lines": n_lines,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "tok_per_sentence": total_tokens / n_lines,
        "tok_per_word": total_tokens / total_words if total_words > 0 else 0.0,
        "tok_per_char": total_tokens / total_chars if total_chars > 0 else 0.0,
        "tok_per_byte": total_tokens / total_bytes if total_bytes > 0 else 0.0,
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
            sem_str = f"{sem_ratio:.2f}x"
            word_str = f"{word_ratio:.2f}x"
            print(f"{lang:<6}{sem_str:<30}{word_str:<25}")


if __name__ == "__main__":
    main()
