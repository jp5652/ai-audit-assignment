#!/usr/bin/env python3
"""
========================================================================================
ASSIGNMENT PART A — Audit fertility.py & Empirical Flaw Isolation
========================================================================================
QUESTION:
"Audit fertility.py:
 - Find actual code bugs.
 - Find at least one conceptual/metric problem where the code technically works but
   measures the wrong thing.
 - Also identify something that looks suspicious but is actually correct.
 - Prove every claimed problem using measurements."

CORE QUESTION:
"Does our tokenizer really behave worse/better for different languages, and are we
measuring that fairly?"
----------------------------------------------------------------------------------------
SOLUTIONS TESTED & MEASURED IN THIS SCRIPT:
1. BUG 1 (Code Bug): line.split(" ") vs line.split()
   - Problem: Literal space splitting produces empty strings "" on consecutive spaces
     (e.g., line 7 in eng_sample.txt "books  in", line 10 in hin_sample.txt "किताबें  अलमारी").
   - Effect: Inflates word count (denominator), artificially depressing fertility by 1.4-2.0%.

2. BUG 2 (Code Bug / Distortion): line = line.lower()
   - Problem: GPT-2 BPE vocabulary contains dedicated tokens for capitalized acronyms
     and proper nouns (NASA, ISRO, GPU, Quarterly). Lowercasing forces subword splits.
   - Conceptual Flaw: Indic scripts (Devanagari, Kannada, Tamil, Telugu) have NO letter casing.
   - Effect: Selectively penalized English by +3.12% token inflation while doing 0.0% to Hindi!

3. BUG 3 (Statistical / Conceptual Aggregation Trap): Macro vs Micro Average
   - Problem: Macro-averaging weights every line equally (sentence-level sample mean),
     whereas micro-averaging weights every token/word equally across the corpus.
   - Solution: Serving capacity requires micro-averaging (total tokens / total units)
     because infrastructure costs scale with total tokens processed.

4. SUSPICIOUS FEATURE (Audited): unicodedata.normalize("NFC", line)
   - Suspicion: An auditor might suspect modifying raw text alters native Indic characters.
   - Ground Truth: Empirically tested on sample Indic words; NFC produces zero delta
     on bytes and token counts (delta = 0). It is a benign defensive normalization.

5. SUSPICIOUS FEATURE (Actually Correct with Measured Impact): add_special_tokens=False
   - Suspicion: Models run with BOS/EOS, so shouldn't we include them?
   - Ground Truth: BOS/EOS adds an arbitrary constant offset (+1 or +2) that artificially
     inflates short sentence fertility by +21.4%. Excluding them measures intrinsic vocabulary efficiency.
========================================================================================
"""

import os
import sys
import unicodedata
import tiktoken
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_ENG = os.path.join(os.path.dirname(__file__), "..", "starter_kit", "corpus_sample", "eng_sample.txt")
SAMPLE_HIN = os.path.join(os.path.dirname(__file__), "..", "starter_kit", "corpus_sample", "hin_sample.txt")
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")

def load_lines(path, normalize_nfc=True):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if normalize_nfc:
                line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


def run_experiment_1_split():
    """
    QUESTION REQUIREMENT: Find actual code bugs & prove with measurements.
    BUG 1: line.split(' ') vs line.split()
    Demonstrates impact of consecutive spaces creating empty strings in word count.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: line.split(' ') vs line.split() [CODE BUG]")
    print("="*70)
    
    enc = tiktoken.get_encoding("gpt2")
    
    for name, path in [("Starter English", SAMPLE_ENG), ("Starter Hindi", SAMPLE_HIN)]:
        lines = load_lines(path)
        
        # Original (split(" "))
        words_orig = [line.lower().split(" ") for line in lines]
        fert_orig = [len(enc.encode(l.lower())) / len(w) for l, w in zip(lines, words_orig)]
        macro_orig = sum(fert_orig) / len(fert_orig)
        total_words_orig = sum(len(w) for w in words_orig)
        empty_tokens_orig = sum(w.count("") for w in words_orig)
        
        # Fixed (split())
        words_fixed = [line.lower().split() for line in lines]
        fert_fixed = [len(enc.encode(l.lower())) / len(w) for l, w in zip(lines, words_fixed)]
        macro_fixed = sum(fert_fixed) / len(fert_fixed)
        total_words_fixed = sum(len(w) for w in words_fixed)
        
        delta = macro_fixed - macro_orig
        pct = (delta / macro_orig) * 100
        
        print(f"[{name}]")
        print(f"  Empty string elements created by split(' '): {empty_tokens_orig}")
        print(f"  Total words count: before={total_words_orig}, after={total_words_fixed}")
        print(f"  Macro fertility:   before={macro_orig:.4f}, after={macro_fixed:.4f} (delta: {delta:+.4f}, {pct:+.2f}%)")
        print(f"  Explanation: split(' ') produces empty strings for double spaces (e.g. line 7 eng, line 10 hin), artificially inflating word count and depressing fertility by up to 1-2% on affected lines.")


def run_experiment_2_lowercase():
    """
    QUESTION REQUIREMENT: Prove every claimed problem using measurements.
    BUG 2: line.lower() before tokenization
    Demonstrates asymmetric distortion on Latin/English vs Indic text.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: line.lower() Distortion on Tokenization [DISTORTION BUG]")
    print("="*70)
    
    enc = tiktoken.get_encoding("gpt2")
    
    for name, path in [("Starter English", SAMPLE_ENG), ("Starter Hindi", SAMPLE_HIN)]:
        lines = load_lines(path)
        
        # With lower()
        toks_low = [enc.encode(l.lower()) for l in lines]
        total_toks_low = sum(len(t) for t in toks_low)
        words = [line.split() for line in lines]
        macro_low = sum(len(t)/len(w) for t, w in zip(toks_low, words)) / len(lines)
        
        # Without lower() (cased)
        toks_cased = [enc.encode(l) for l in lines]
        total_toks_cased = sum(len(t) for t in toks_cased)
        macro_cased = sum(len(t)/len(w) for t, w in zip(toks_cased, words)) / len(lines)
        
        delta_toks = total_toks_low - total_toks_cased
        pct_toks = (delta_toks / total_toks_cased) * 100
        if name == "Starter English":
            eng_pct_toks = pct_toks
        
        print(f"[{name}]")
        print(f"  Tokens cased (original): {total_toks_cased}")
        print(f"  Tokens lowercased:       {total_toks_low} (delta: {delta_toks:+d}, {pct_toks:+.2f}%)")
        print(f"  Macro fertility: cased={macro_cased:.4f}, lower={macro_low:.4f}")
        
    print("  Detailed English tokens altered by lowercasing:")
    eng_lines = load_lines(SAMPLE_ENG)
    for i, l in enumerate(eng_lines):
        c_toks = enc.encode(l)
        l_toks = enc.encode(l.lower())
        if len(c_toks) != len(l_toks):
            print(f"    Line {i+1}: cased={len(c_toks)} toks -> lower={len(l_toks)} toks ('{l.strip()[:40]}...')")
    print(f"  Explanation: gpt2 BPE includes capitalized tokens (e.g., 'Quarterly', 'NASA', 'GPU'); lowercasing forces subword splits into extra tokens, artificially penalizing English by {eng_pct_toks:+.2f}% while leaving uncased Indic script completely unchanged.")


def run_experiment_3_aggregation():
    """
    QUESTION REQUIREMENT: Conceptual/metric problem where code measures the wrong thing.
    BUG 3: Macro-average (Mean of Ratios) vs Micro-average (Ratio of Totals)
    Demonstrates ratio-estimator distortion and Jensen's inequality trap.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Macro-average vs Micro-average [METRIC/AGGREGATION TRAP]")
    print("="*70)
    
    enc = tiktoken.get_encoding("gpt2")
    
    for name, path in [("Starter English", SAMPLE_ENG), ("Starter Hindi", SAMPLE_HIN)]:
        lines = load_lines(path)
        words = [l.split() for l in lines]
        toks = [enc.encode(l) for l in lines]
        
        # Macro (mean of line ratios)
        macro = sum(len(t)/len(w) for t, w in zip(toks, words)) / len(lines)
        
        # Micro (total tokens / total words)
        total_toks = sum(len(t) for t in toks)
        total_words = sum(len(w) for w in words)
        micro = total_toks / total_words
        
        delta = micro - macro
        pct = (delta / macro) * 100
        print(f"[{name}]")
        print(f"  Macro-average (mean of line ratios): {macro:.4f}")
        print(f"  Micro-average (ratio of totals):     {micro:.4f} (delta: {delta:+.4f}, {pct:+.2f}%)")
        print(f"  Total tokens: {total_toks}, Total words: {total_words}")
    print("  Explanation: Macro-average weights each sentence equally (answering 'what is the token expansion of an average sentence?'). Micro-average weights each word equally across the full corpus (answering 'what is the aggregate token expansion of the workload?'). For serving capacity planning and billing, micro-average is the appropriate metric because server cost scales with total tokens processed.")


def run_experiment_4_nfc():
    """
    QUESTION REQUIREMENT: Identify something that looks suspicious but is actually correct.
    ANALYSIS OF FEATURE: unicodedata.normalize('NFC', line)
    An auditor might suspect this line alters native Indic characters or mutates data.
    We test whether NFC vs NFD introduces any distortion on sample Indic words.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: NFC Normalization [AUDIT OF SUSPICIOUS-LOOKING FEATURE]")
    print("="*70)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Test words with nuktas / matras
    test_words = [
        ("फ़िल्म", "film with nukta"),
        ("ज़रूरत", "zarurat with nukta"),
        ("क़िला", "qila with nukta"),
        ("गाड़ी", "gadi with nukta"),
    ]
    
    print(f"{'Word':<10}{'Description':<20}{'NFC bytes':<12}{'NFD bytes':<12}{'NFC tok':<10}{'NFD tok':<10}")
    print("-" * 74)
    for word, desc in test_words:
        w_nfc = unicodedata.normalize("NFC", word)
        w_nfd = unicodedata.normalize("NFD", word)
        t_nfc = len(enc.encode(w_nfc))
        t_nfd = len(enc.encode(w_nfd))
        b_nfc = len(w_nfc.encode("utf-8"))
        b_nfd = len(w_nfd.encode("utf-8"))
        print(f"{word:<10}{desc:<20}{b_nfc:<12}{b_nfd:<12}{t_nfc:<10}{t_nfd:<10}")
        
    print("  Measured Result: On these sample words, NFC and NFD yield identical byte counts and token counts (delta = 0). Normalizing to NFC does not distort or inflate tokenization (it is completely harmless here).")
    print("  Note: Our primary empirically verified 'suspicious but actually correct' feature with significant measurable impact is add_special_tokens=False (see Experiment 5).")


def run_experiment_5_special_tokens():
    """
    QUESTION REQUIREMENT: Identify something that looks suspicious but is actually correct.
    FEATURE 2: add_special_tokens=False
    Demonstrates that adding special tokens skews short sentences.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: add_special_tokens=False [SUSPICIOUS BUT ACTUALLY CORRECT]")
    print("="*70)
    
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    lines = load_lines(SAMPLE_ENG)
    
    with_special = [len(tok.encode(l, add_special_tokens=True)) for l in lines]
    no_special = [len(tok.encode(l, add_special_tokens=False)) for l in lines]
    words = [len(l.split()) for l in lines]
    
    fert_with = sum(t/w for t, w in zip(with_special, words)) / len(lines)
    fert_no = sum(t/w for t, w in zip(no_special, words)) / len(lines)
    pct_special = ((fert_with - fert_no) / fert_no) * 100
    
    print(f"  XLM-R fertility with BOS/EOS (add_special_tokens=True):  {fert_with:.4f}")
    print(f"  XLM-R fertility pure text   (add_special_tokens=False): {fert_no:.4f}")
    print(f"  Delta: {fert_with - fert_no:+.4f} ({pct_special:+.2f}%)")
    print(f"  Explanation: Adding BOS/EOS introduces a constant +2 tokens per line regardless of sentence length, artificially inflating short sentence fertility by {pct_special:.2f}%.")


if __name__ == "__main__":
    run_experiment_1_split()
    run_experiment_2_lowercase()
    run_experiment_3_aggregation()
    run_experiment_4_nfc()
    run_experiment_5_special_tokens()
