#!/usr/bin/env python3
"""
========================================================================================
ASSIGNMENT PART A — Multilingual Evaluation Corpus Construction & Provenance
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
1. Source & Provenance: Meta NLLB FLORES-200 benchmark (devtest split).
   Direct URL: https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
2. Languages selected (5 languages, exceeding the 4-language requirement):
   - eng (English, Latin script) - High-resource Germanic baseline
   - hin (Hindi, Devanagari script) - Indo-Aryan language (Northern India)
   - kan (Kannada, Kannada script) - Dravidian language (Karnataka)
   - tam (Tamil, Tamil script) - Dravidian language (Tamil Nadu)
   - tel (Telugu, Telugu script) - Dravidian language (Andhra Pradesh / Telangana)
3. Size: Exactly 1,012 parallel sentences per language (5,060 sentences total).
4. Preprocessing: Canonical Unicode NFC normalization (unicodedata.normalize('NFC', line)).
5. What the corpus CANNOT tell you (Caveat):
   - FLORES-200 consists of formal news/Wikipedia articles in standard native orthography.
   - It cannot evaluate performance on real-world conversational chat, slang, typos,
     or code-mixed / Latin-transliterated text (e.g. Romanized Hinglish or Kanglish).
   - A sample size of 1,012 clean translated sentences does not capture long-tail
     production distribution shifts, dialectal variations, or rare domain vocabularies.
========================================================================================
"""

import hashlib
import os
import sys
import tarfile
import unicodedata
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR = os.path.join(os.path.dirname(__file__), "corpus", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "corpus")
TARBALL_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

LANG_MAP = {
    "eng": "eng_Latn.devtest",
    "hin": "hin_Deva.devtest",
    "kan": "kan_Knda.devtest",
    "tam": "tam_Taml.devtest",
    "tel": "tel_Telu.devtest",
}


def ensure_raw_files_exist():
    """Ensures raw FLORES-200 files exist, downloading from official source if missing."""
    os.makedirs(RAW_DIR, exist_ok=True)
    missing = [fname for fname in LANG_MAP.values() if not os.path.exists(os.path.join(RAW_DIR, fname))]
    if not missing:
        return

    print(f"Missing raw FLORES files ({missing}). Downloading archive from Meta public repository...")
    tar_path = os.path.join(RAW_DIR, "flores200.tar.gz")
    urllib.request.urlretrieve(TARBALL_URL, tar_path)

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            for fname in LANG_MAP.values():
                if member.name.endswith(fname):
                    extracted = tar.extractfile(member)
                    target = os.path.join(RAW_DIR, fname)
                    with open(target, "wb") as out:
                        out.write(extracted.read())
                    print(f"  Extracted: {fname}")

    if os.path.exists(tar_path):
        os.remove(tar_path)


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:12]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ensure_raw_files_exist()

    line_counts = {}
    stats = {}

    print(f"{'Lang':<6}{'Sentences':<12}{'Words (split)':<15}{'Chars':<12}{'UTF-8 Bytes':<14}{'SHA-256 (12char)':<18}")
    print("-" * 78)

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

        if len(lines) == 0:
            raise ValueError(f"Corpus file {fname} is empty!")

        with open(out_path, "w", encoding="utf-8") as f:
            for l in lines:
                f.write(l + "\n")

        line_counts[lang] = len(lines)
        chk = compute_sha256(out_path)

        stats[lang] = {
            "num_lines": len(lines),
            "num_words": total_words,
            "num_chars": total_chars,
            "num_bytes": total_bytes,
            "checksum": chk,
        }

        print(
            f"{lang:<6}{len(lines):<12}{total_words:<15}{total_chars:<12}{total_bytes:<14}{chk:<18}"
        )

    # Verification: Assert parallel alignment across all languages
    unique_lengths = set(line_counts.values())
    assert len(unique_lengths) == 1, f"Parallel alignment error: Mismatched sentence counts {line_counts}"
    print("-" * 78)
    print(f"Alignment verified: All {len(LANG_MAP)} languages have exactly {list(unique_lengths)[0]} parallel sentences.")
    print(f"Corpus assembled at {OUT_DIR}")


if __name__ == "__main__":
    main()
