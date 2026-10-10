# Feature Extraction Pipeline Report

**Generated**: `2026-10-10T21:51:16.586738`  
**Total Features Extracted**: `72`  
**Train / Test Feature Schema Match**: `PASSED`  
**Target & Identifier Exclusion**: `PASSED (Zero Leakage)`  

---

## 1. Processed Samples Overview

| Split | Processed | Valid (`ok`) | Failed | Missing Feature Values |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 769 | 769 | 0 | 0 |
| **Test** | 216 | 216 | 0 | 0 |
| **Total** | **985** | **985** | **0** | **0** |

---

## 2. Feature Categories & Definitions

The 72 features span 5 distinct complementary domains:

1. **Linguistic Length & Statistics (9)**: Word count, sentence count, char count, avg word length, avg/min/max/std sentence lengths.
2. **Lexical Diversity & Breadth (6)**: Long word count & ratio ($\ge 7$ chars), unique words, Type-Token Ratio (TTR), Root TTR (Guiraud's Index), Herdan's log TTR.
3. **Disfluency & Hesitation Signals (10)**: Stop words ratio, immediately repeated words ratio, overt fillers (`um`, `uh`), discourse markers (`like`, `you know`), total filler rate.
4. **Grammar & Syntactic Complexity (30)**:
   - Punctuation & clause boundary ratios (commas, periods, question marks).
   - Personal, possessive, and relative pronouns.
   - Auxiliary verbs & modal verbs.
   - Subordinating vs. coordinating conjunctions & subordination index.
   - Preposition counts and proportions.
   - Morphological derivational suffixes (nominalizations `-tion`, `-ment`; adjectives `-able`, `-ive`; adverbs `-ly`; past tense `-ed`; progressive `-ing`).
5. **Acoustic & Multimodal Fluency (17)**: Audio duration, RMS energy dynamics, zero-crossing rates, silence & speech ratios, words-per-minute (WPM), phonation rate, and articulation pace.

---

## 3. Constant or Low-Variance Features
- **Constant Features Identified**: `['exclamation_count']`

---

## 4. NLP Dependency Note
- **External POS Models (e.g. Spacy/NLTK)**: `Not Installed / Bypassed`.
- Feature extraction utilizes deterministic, CPU-efficient rule-based parsing and curated lexicons. No heavyweight downloads or external model dependencies are required.
