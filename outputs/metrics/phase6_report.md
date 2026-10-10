# Phase 6: Model Training and Validation Report

**Generated**: `2026-10-10T21:52:42.997700`  
**Training Samples**: `769` (Split: `615` Train / `154` Val, Seed: `42`)  
**Best Model Candidate**: `Extra Trees`  
**Compulsory Training RMSE**: `0.4808`  
**Validation RMSE**: `0.7683`  
**Validation Pearson $r$**: `0.8270`  

---

## 1. Holdout Model Comparison (80% Train / 20% Val)

All models evaluated with strictly leakage-free preprocessing pipelines (imputation and scaling fitted strictly on training data). Post-processing clips predictions to `[0.0, 5.0]`.

| Model | Train RMSE | Val RMSE | Val Pearson $r$ | Val MAE | Val $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees** | 0.4808 | 0.7683 | 0.8270 | 0.6117 | 0.6782 |
| **Random Forest** | 0.4652 | 0.7736 | 0.8227 | 0.6134 | 0.6737 |
| **HistGradientBoosting** | 0.2397 | 0.7834 | 0.8179 | 0.6114 | 0.6654 |
| **Ridge Regression** | 0.7732 | 0.8132 | 0.8004 | 0.6345 | 0.6395 |
| **Dummy (Mean Baseline)** | 1.2062 | 1.3596 | nan | 1.0638 | -0.0078 |

---

## 2. 5-Fold Cross-Validation Summary

Cross-validation evaluated across all 769 training samples (Mean $\pm$ Std):

| Model | CV Train RMSE | CV Val RMSE | CV Val Pearson $r$ | CV Val MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy (Mean Baseline)** | 1.238 $\pm$ 0.017 | 1.240 $\pm$ 0.063 | nan $\pm$ nan | 0.998 $\pm$ 0.035 |
| **Ridge Regression** | 0.769 $\pm$ 0.012 | 0.853 $\pm$ 0.062 | 0.713 $\pm$ 0.079 | 0.678 $\pm$ 0.056 |
| **Random Forest** | 0.457 $\pm$ 0.009 | 0.807 $\pm$ 0.044 | 0.749 $\pm$ 0.058 | 0.641 $\pm$ 0.048 |
| **Extra Trees** | 0.480 $\pm$ 0.010 | 0.808 $\pm$ 0.055 | 0.748 $\pm$ 0.068 | 0.648 $\pm$ 0.057 |
| **HistGradientBoosting** | 0.255 $\pm$ 0.015 | 0.831 $\pm$ 0.049 | 0.734 $\pm$ 0.063 | 0.661 $\pm$ 0.049 |

---

## 3. Key Observations & Model Selection
- **Target Leakage**: Zero. Sample IDs, split indicators, and test records were strictly segregated.
- **Baseline Comparison**: All machine learning models comfortably outperform the naive `Dummy (Mean Baseline)` predictor on both RMSE and Pearson $r$.
- **Best Candidate**: Selected based on the lowest holdout Validation RMSE while maintaining high positive Pearson correlation with actual human grammar ratings.
