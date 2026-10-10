# 🎙️ Grammar Scoring Engine — SHL Research Engineer Assessment (2026)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-69%20passed-success.svg)](https://pytest.org/)
[![Evaluation Pearson r](https://img.shields.io/badge/Val%20Pearson%20r-0.8270-brightgreen.svg)]()
[![Evaluation RMSE](https://img.shields.io/badge/Val%20RMSE-0.7683-green.svg)]()

An end-to-end Machine Learning pipeline that predicts continuous English grammar proficiency scores (**0.0 to 5.0**) from raw ~45–60 second spoken voice recordings (`.wav`), developed for the **SHL Research Engineer hiring challenge**.

---

## 📌 Executive Summary & Architecture

```text
       ┌────────────────────────┐
       │   Raw Voice Audio      │ (~45-60s WAV, 16kHz mono)
       └───────────┬────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│ Whisper ASR     │ │ Acoustic Engine │
│ (base.en, FP16) │ │ (librosa)       │
└────────┬────────┘ └────────┬────────┘
         │                   │
         ▼                   ▼
┌─────────────────────────────────────┐
│ Multi-Modal Feature Extraction      │ (72 numerical signals)
│ • Lexical Diversity (TTR, Root-TTR) │ • Syntactic Depth & Structure
│ • Disfluency & Discourse Fillers    │ • Heuristics (Tense, Affixes)
│ • Acoustic Fluency & Speaking Rate  │ • Phonation & Pause Ratios
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Median Imputation & Standardization │ (Inside Pipeline, Zero Leakage)
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Extra Trees Ensemble Regressor      │ (500 Estimators, Seed 42)
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Continuous Score Output [0.0 - 5.0] │
└─────────────────────────────────────┘
```

The system combines state-of-the-art automatic speech recognition (`openai/whisper-base.en`) with acoustic speech signal processing to extract **72 interpretable, multimodal features**. These features feed into a strictly validated ensemble regressor (`ExtraTreesRegressor`), trained with zero data leakage.

---

## 📊 Evaluation & Model Performance

### 1. Final Model Metrics (`Extra Trees`)

- **Full In-Sample Training RMSE**: **`0.5144`**
- **In-Sample Training Pearson $r$**: **`0.9185`**
- **In-Sample Training MAE**: **`0.4091`**
- **In-Sample Training $R^2$**: **`0.8274`**
- **Independent 20% Holdout Validation RMSE**: **`0.7683`**
- **Independent 20% Holdout Validation Pearson $r$**: **`0.8270`**
- **5-Fold Stratified Cross-Validation RMSE**: **`0.8084 ± 0.0550`**
- **5-Fold Stratified Cross-Validation Pearson $r$**: **`0.7479 ± 0.0675`**

### 2. Multi-Model Comparison (80/20 Holdout Split)

| Regressor Pipeline | Train RMSE | Val RMSE | Val Pearson $r$ | Val MAE | Val $R^2$ | 5-Fold CV Val RMSE | 5-Fold CV Pearson $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees (Selected)** | **0.4808** | **0.7683** | **0.8270** | **0.6117** | **0.6782** | **0.8084 ± 0.055** | **0.7479 ± 0.068** |
| Random Forest | 0.4652 | 0.7736 | 0.8227 | 0.6134 | 0.6737 | 0.8070 ± 0.044 | 0.7491 ± 0.058 |
| HistGradientBoosting | 0.2397 | 0.7834 | 0.8179 | 0.6114 | 0.6654 | 0.8309 ± 0.049 | 0.7336 ± 0.063 |
| Ridge Regression (Scaled) | 0.7732 | 0.8132 | 0.8004 | 0.6345 | 0.6395 | 0.8533 ± 0.062 | 0.7125 ± 0.079 |
| Dummy Baseline (Mean) | 1.2062 | 1.3596 | N/A | 1.0638 | -0.0078 | 1.2397 ± 0.063 | N/A |

---

## 🔍 Feature Engineering Methodology (72 Features)

The 72 features are organized into six cohesive linguistic and acoustic categories:

1. **Volume & Length (12 features)**: Word count, sentence count, raw/nospace character counts, average/min/max/std sentence lengths, long word count and ratio.
2. **Lexical Diversity (3 features)**: Type-Token Ratio (`TTR`), Root-TTR, and Bilogarithmic TTR (`Log-TTR`).
3. **Disfluency & Fillers (6 features)**: Filler word frequencies (e.g., *um*, *uh*, *like*, *you know*), discourse markers, and speaking rate normalized filler rates.
4. **Punctuation & Syntax (7 features)**: Punctuation density, clause boundary ratio, commas, periods, question marks, and exclamation marks.
5. **Part-of-Speech & Grammatical Markers (31 features)**: Pronouns (personal, possessive, relative), auxiliary and modal verbs, subordinating/coordinating conjunctions, subordination index, prepositions, and morphological affixes (noun `-ment`, `-tion`, adjective `-able`, adverb `-ly`, past tense `-ed`, progressive `-ing`).
6. **Acoustic Fluency & Dynamics (13 features)**: Energy RMS (mean, std, peak), Zero Crossing Rate (`ZCR`), silence ratio, speech ratio, speech duration, speaking rate (words/sec, words/min), phonation rate, and articulation pace.

---

## 📁 Repository Structure

```text
shl-hiring-assessment/
├── app.py                     # Flask interactive web demo application
├── requirements.txt           # Verified Python dependencies
├── README.md                  # Comprehensive documentation and results
├── .gitignore                 # Excludes raw audio, large caches, and model weights
├── data/
│   ├── cache/                 # Whisper transcript caches (.gitkeep)
│   ├── processed/             # Extracted train & test feature matrices (.gitkeep)
│   └── raw/                   # Raw audio directory (.gitkeep)
├── models/
│   ├── best_model.joblib      # Holdout-evaluated model artifact
│   └── final_grammar_scorer.joblib # Final production Extra Trees pipeline
├── notebooks/
│   └── grammar_scoring_engine.ipynb # Reproducible competition Jupyter Notebook
├── outputs/
│   ├── metrics/               # Evaluation JSONs, CSV comparison tables
│   ├── plots/                 # Actual vs. predicted, residual distributions, heatmaps
│   └── predictions/           # Generated test predictions and submission CSVs
├── src/
│   ├── __init__.py
│   ├── audio.py               # Robust audio loader, resampler, silence detector
│   ├── config.py              # Centralized configuration & paths
│   ├── data.py                # Dataset validation and loading
│   ├── features.py            # 72-feature multimodal extraction engine
│   ├── models.py              # Scikit-learn pipelines & cross-validation
│   ├── transcribe.py          # Whisper FP16 transcription pipeline with caching
│   └── utils.py               # Deterministic seeding, logging, timer utilities
├── static/                    # Frontend CSS and JS for web demo
├── templates/
│   └── index.html             # UI template for audio scoring web interface
└── tests/                     # 69 automated unit and integration tests
```

---

## 🚀 Getting Started & Reproduction

### 1. Environment Setup

```powershell
# Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd shl-hiring-assessment

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA (or CPU)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite

```powershell
python -m pytest tests/ -v
```
*(All 69 unit and regression tests pass).*

### 3. Launch Interactive Web Demo

Test any voice recording and receive an instant grammar score (0.0 to 5.0) and analysis:

```powershell
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 4. Run the Competition Jupyter Notebook

Open and run `notebooks/grammar_scoring_engine.ipynb` to inspect all stages of EDA, feature extraction, cross-validation, and diagnostic plots.

---

## 📝 Submission Files

- **`outputs/predictions/submission_test_216.csv`**: Contains verified predictions for all 216 test records in `filename,label` format matching `test.csv`.
- **`outputs/predictions/test_predictions.csv`**: Detailed predictions with accompanying metadata (audio duration, word count, speaking rate).

---

## ⚖️ License & Confidentiality

Developed exclusively for the **SHL Research Engineer Hiring Challenge (2026)**. Model architecture, feature engineering pipelines, and source code are open for academic and evaluation review. Raw candidate audio recordings and competition dataset files are excluded in accordance with competition rules.
