# Test Predictions & Submission Format Diagnosis Report

**Generated**: `2026-10-10T22:06:43.387708`  
**Test Records Processed**: `216`  
**Predicted Score Range**: `[2.3957, 4.6930]` (Mean: `3.3606`, Std: `0.5194`)  

---

## 1. Submission Template Discrepancy Analysis

An audit of the competition files reveals an inconsistency between `test.csv` and `sample_submission.csv`:

| Property | `test.csv` | `sample_submission.csv` | Status / Observation |
| :--- | :---: | :---: | :--- |
| **Row Count** | **216** | **204** | **Mismatch**: Difference of 12 rows |
| **Physical WAV Files in `test/`** | **216** | **25** (only 25 match) | `test.csv` matches `test/` audio folder **100%** |
| **Filenames found in `train/`** | 0 | **105** | 105 files in `sample_submission.csv` are training files |
| **Filenames not in dataset** | 0 | **99** | 99 files in `sample_submission.csv` do not exist in dataset |

---

## 2. Recommendation & Safe Resolution Policy

1. **Primary Ground Truth**: `test.csv` is the verified source of truth for test evaluation. It contains exactly 216 records corresponding to the 216 test WAV audio files on disk (`audio_128.wav`, etc.).
2. **Provisional Predictions Saved**:
   - `outputs/predictions/test_predictions.csv`: 216 records with rich metadata (`filename`, `predicted_score`, `audio_duration_s`, `word_count`, `speaking_rate_wpm`).
   - `outputs/predictions/submission_test_216.csv`: 216 records strictly formatted as `filename,label` matching `test.csv`.
3. **Action Required Before Final Portal Submission**:
   - Verify with the competition portal/organizers whether the submission portal expects the 216 records matching `test.csv` or if they provided a corrected submission template.
