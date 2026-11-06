# GAMBIt+ Dataset — COMET-Kiwi Baseline

This pipeline analyzes the GAMBIt+ dataset, calculates COMET-Kiwi quality estimation scores for masculine and feminine translations, and merges the results into a single Parquet file.

## Pipeline Overview

```text
GAMBIt+ Dataset (Hugging Face)
        |
        v
1. analyze_gambit_dataset.py
        |
        v
2. score_gambit_plus_with_cometkiwi.py
        |
        v
3. merge_target_scores_per_source.py
        |
        v
4. merge_all_source_scores.py
        |
        v
5. score_analysis.py
```

## 1. Dataset Analysis

Run `analyze_gambit_dataset.py` to explore the GAMBIt+ dataset.

The script provides general dataset statistics, language-pair distributions, text-length statistics, and a sample record.

```powershell
python analyze_gambit_dataset.py
```

## 2. COMET-Kiwi Scoring

Run `score_gambit_plus_with_cometkiwi.py` to calculate COMET-Kiwi scores for a specific source-target language pair.

**Example: English → Greek (en-el)**

```powershell
python score_gambit_plus_with_cometkiwi.py `
    --source_language en `
    --target_language el `
    --batch_size 8 `
    --output_dir "../../data/train_data/baseline_scores"
```

The script calculates the following scores:

- `masc_score`: Quality estimation score for the masculine translation.
- `fem_score`: Quality estimation score for the feminine translation.
- `score_diff`: Difference between masculine and feminine scores.
- `score_mean`: Average of the two scores.
- `abs_score_diff`: Absolute difference between the two scores.

**Output:**

`gambit_en-el_cometkiwi.parquet`

Repeat the scoring process for all required language pairs.

Source languages: `en`, `fi`, `tr`

Target languages: `ar`, `cs`, `el`, `es`, `fr`, `is`, `it`, `pt`, `ru`, `sr`, `uk`

## 3. Merge Target Scores per Source

Run `merge_target_scores_per_source.py` to combine all target-language Parquet files for one source language.

**Example: English**

```powershell
python merge_target_scores_per_source.py `
    --source_language en `
    --output_dir "../../data/train_data/baseline_scores"
```

**Output:**

`gambit_en_all_targets_cometkiwi.parquet`

Repeat for Finnish and Turkish:

```powershell
python merge_target_scores_per_source.py `
    --source_language fi `
    --output_dir "../../data/train_data/baseline_scores"

python merge_target_scores_per_source.py `
    --source_language tr `
    --output_dir "../../data/train_data/baseline_scores"
```

All target-language score files must exist before merging each source language.

## 4. Merge All Source Scores

Run `merge_all_source_scores.py` to combine the merged English, Finnish, and Turkish datasets into a single Parquet file.

```powershell
python merge_all_source_scores.py `
    --output_dir "../../data/train_data/baseline_scores"
```

**Final output:**

`gambit_all_sources_all_targets_cometkiwi.parquet`

This file contains the combined COMET-Kiwi baseline scores across all processed language pairs.

## 5. Score Analysis

Run `score_analysis.py` to examine the baseline scores and gender-related scoring differences.

```powershell
python score_analysis.py `
    --input_file "../../data/train_data/baseline_scores/gambit_all_sources_all_targets_cometkiwi.parquet"
```

The analysis includes descriptive score statistics, score direction, language-pair comparisons, and the largest absolute score differences.

## Notes

- All commands assume execution from `project/data_processing/gambit+/` using PowerShell.
- Configuration paths and supported languages are defined in the project's root `config.py`.
- The scripts use the `BASELINE_SCORES_DIR` configuration to locate input files for merging.
- Scoring can run on CPU or CUDA GPU, depending on hardware availability.
- The complete scoring process must finish before running the merge scripts.