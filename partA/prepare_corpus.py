#!/usr/bin/env python3
"""
prepare_corpus.py -- Prepares multilingual parallel evaluation corpus from FLORES-200.

Languages selected:
- eng: English (eng_Latn) - High-resource Germanic/Indo-European baseline
- hin: Hindi (hin_Deva) - Indo-Aryan / Indo-European language (Northern India)
- kan: Kannada (kan_Knda) - Dravidian language (Karnataka)
- tam: Tamil (tam_Taml) - Dravidian language (Tamil Nadu)
- tel: Telugu (tel_Telu) - Dravidian language (Andhra Pradesh / Telangana)

Split: FLORES-200 devtest (1,012 parallel sentences per language).
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
