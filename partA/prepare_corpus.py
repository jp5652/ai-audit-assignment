#!/usr/bin/env python3
"""
========================================================================================
ASSIGNMENT PART A — Multilingual Evaluation Corpus Construction
========================================================================================
QUESTION:
"Build a proper multilingual evaluation corpus with at least:
  - English
  - Hindi
  - 2 Dravidian languages such as Kannada, Tamil, Telugu, or Malayalam.
Document corpus size, domain, preprocessing, and what your corpus cannot tell you."

CORE QUESTION:
"Does our tokenizer really behave worse/better for different languages, and are we
measuring that fairly?"
----------------------------------------------------------------------------------------
SOLUTION IMPLEMENTED BELOW:
1. Source: We assemble the evaluation corpus from Meta's official NLLB FLORES-200 benchmark
   (devtest split), which provides sentence-aligned, human-translated parallel text.
2. Languages selected (5 languages, exceeding the 4-language requirement):
   - eng (English, Latin script) - High-resource Germanic baseline
   - hin (Hindi, Devanagari script) - Indo-Aryan language (Northern India)
   - kan (Kannada, Kannada script) - Dravidian language (Karnataka)
   - tam (Tamil, Tamil script) - Dravidian language (Tamil Nadu)
   - tel (Telugu, Telugu script) - Dravidian language (Andhra Pradesh / Telangana)
3. Size: Exactly 1,012 parallel sentences per language (5,060 sentences total).
4. Preprocessing: Canonical Unicode NFC normalization (unicodedata.normalize('NFC', line)).
5. What the corpus CANNOT tell you (Caveat):
   FLORES-200 consists of formal news/Wikipedia articles in standard native orthography.
   It cannot evaluate performance on real-world conversational chat, slang, typos,
   or code-mixed / Latin-transliterated text (e.g. Romanized Hinglish or Kanglish).
========================================================================================
"""

import os
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR = os.path.join(os.path.dirname(__file__), "corpus", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "corpus")

LANG_MAP = {
    "eng": "eng_Latn.devtest",
    "hin": "hin_Deva.devtest",
    "kan": "kan_Knda.devtest",
    "tam": "tam_Taml.devtest",
    "tel": "tel_Telu.devtest",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = {}

    print(f"{'Lang':<6}{'Sentences':<12}{'Words (split)':<15}{'Chars':<12}{'UTF-8 Bytes':<14}")
    print("-" * 60)

    for lang, fname in LANG_MAP.items():
        in_path = os.path.join(RAW_DIR, fname)
        out_path = os.path.join(OUT_DIR, f"{lang}.txt")

        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Missing raw file: {in_path}")

        lines = []
        total_words = 0
        total_chars = 0
        total_bytes = 0

        with open(in_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                # Normalize to NFC
                line = unicodedata.normalize("NFC", line)
                lines.append(line)
                total_words += len(line.split())
                total_chars += len(line)
                total_bytes += len(line.encode("utf-8"))

        with open(out_path, "w", encoding="utf-8") as f:
            for l in lines:
                f.write(l + "\n")

        stats[lang] = {
            "num_lines": len(lines),
            "num_words": total_words,
            "num_chars": total_chars,
            "num_bytes": total_bytes,
        }

        print(
            f"{lang:<6}{len(lines):<12}{total_words:<15}{total_chars:<12}{total_bytes:<14}"
        )

    print("-" * 60)
    print(f"Corpus successfully assembled in {OUT_DIR}")


if __name__ == "__main__":
    main()
