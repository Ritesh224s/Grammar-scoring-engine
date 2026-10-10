# Dataset Ingestion & Validation Summary

**Execution Time**: `2026-10-10T17:23:49.936975`  
**Data Directory**: `C:\Users\aacha\Downloads\shl-hiring-assessment-2026\Dataset_Final`  
**Validation Status**: **PASSED_WITH_WARNINGS**

---

## 1. Dataset Counts & Matching

| Split | CSV Rows | Audio Dir WAVs | Matched Audio | Missing Audio | Corrupt WAVs | Duration (min/mean/max) |
|---|---|---|---|---|---|---|
| **Train** | 769 | 769 | 769 | 0 | 0 | 20.06s / 55.78s / 61.04s |
| **Test** | 216 | 216 | 216 | 0 | 0 | 6.48s / 48.76s / 61.03s |
| **Sample Sub** | 204 | N/A | N/A | N/A | N/A | N/A |

---

## 2. Training Target Label Statistics

- **Total Labels**: 769
- **Mean Score**: 3.3134
- **Standard Deviation**: 1.239
- **Score Range**: [0.0, 5.0]
- **Quartiles (25% / 50% / 75%)**: 2.5 / 3.0 / 4.0
- **Missing / Invalid Labels**: 0 missing, 0 out of bounds

---

## 3. Filename Overlap Analysis

- **Train/Test Shared Filenames**: 212 files share identical names (e.g. `audio_0.wav` ... `audio_215.wav`).
- **Collision Safeguard**: Every sample is assigned a globally unique `sample_id` (`train/<filename>` and `test/<filename>`). Audio paths point to separate directories.

---

## 4. Submission Mismatch Investigation

- `test.csv` rows: **216** (`audio_0.wav` to `audio_215.wav`)
- `sample_submission.csv` rows: **204**
- **Overlapping IDs between Test and Sample Submission**: 25
- **IDs in Sample Submission that belong to Train**: 105
- **Diagnosis**: Mismatch detected: sample_submission.csv (204 rows) appears to be from a prior/draft split. test.csv (216 rows, audio_0.wav to audio_215.wav) represents the true evaluation set. Safe resolution: Generate predictions for all 216 test.csv rows in ['filename', 'label'] schema.

---

## 5. Audit Warnings & Errors

### Warnings (3)
- [WARNING] test.csv contains 'label' column with values [-1]. These are dummy placeholders and must never be treated as ground truth.
- [WARNING] 212 audio filenames appear in BOTH train.csv and test.csv (e.g. 'audio_0.wav'). Handled safely: features and paths are indexed by unique sample_id ('train/...' vs 'test/...').
- [WARNING] Submission mismatch: test.csv has 216 rows while sample_submission.csv has 204 rows. Prediction pipeline will output all 216 test samples.

### Errors (0)
- None
