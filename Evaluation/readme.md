# QE Evaluation Pipeline

This folder contains the evaluation pipeline for comparing **pretrained and fine-tuned Machine Translation Quality Estimation (QE) models**.

The pipeline scores machine translations, computes evaluation statistics, and generates comparison reports.

Currently supported metric: **COMET-Kiwi 22** (`Unbabel/wmt22-cometkiwi-da`).

## 1. Project Structure

```text
evaluation/
├── datasets/             # Load and validate datasets
├── evaluators/           # Run model evaluation
├── qe_metrics/           # Load QE models and compute scores
├── statistic_analysis/   # Calculate evaluation statistics
├── reporting/            # Generate comparison tables
├── utils/                # Shared utility functions
├── evaluate_dataset.py   # Main evaluation script
└── generate_report.py    # Report generation script
```

## 2. Evaluation Datasets

The evaluation uses **three categories of datasets**, handled by two evaluation modes: `general` and `gender`.

All input datasets must be in **Parquet format**.

| Category | Dataset | Evaluation mode |
|---|---|---|
| General quality evaluation | MLQE-PE | `general` |
| Gender-ambiguous | GATE | `gender` |
| Gender-unambiguous | MT-GenEval Counterfactual | `gender` |

### General Quality Evaluation

Used to evaluate the overall quality estimation performance of the model.

Required Parquet columns:

| Column | Description |
|---|---|
| `source` | Original source sentence |
| `lang_pair` | Language pair |
| `mt` | Machine translation |
| `score` | Optional ground-truth quality score |

If `score` is available, Pearson and Spearman correlations are calculated.

### Gender Evaluation

Used to measure whether the QE model assigns different scores to masculine and feminine translation variants.

Required Parquet columns:

| Column | Description |
|---|---|
| `source` | Original source sentence |
| `lang_pair` | Language pair |
| `masc` | Masculine translation |
| `fem` | Feminine translation |

Both gender-ambiguous and gender-unambiguous datasets use the same evaluation pipeline.

Additional dataset columns are preserved in the output.

## 3. Running the Evaluation

Run the scripts from the `evaluation` directory.

```powershell
cd .\evaluation
```

Install the required dependencies:

```powershell
python -m pip install torch unbabel-comet pandas pyarrow numpy scipy
```

### General Evaluation

Example using MLQE-PE:

```powershell
python evaluate_dataset.py `
    --data_file "PATH_TO/mlqe_pe_test.parquet" `
    --evaluation_type general `
    --dataset_name MLQE-PE `
    --metric_name cometkiwi `
    --output_dir results
```

### Gender-Ambiguous Evaluation

Example using GATE:

```powershell
python evaluate_dataset.py `
    --data_file "PATH_TO/gate.parquet" `
    --evaluation_type gender `
    --dataset_name GATE `
    --metric_name cometkiwi `
    --output_dir results
```

### Gender-Unambiguous Evaluation

Example using MT-GenEval Counterfactual:

```powershell
python evaluate_dataset.py `
    --data_file "PATH_TO/mt_geneval_counterfactual.parquet" `
    --evaluation_type gender `
    --dataset_name MT-GenEval-Counterfactual `
    --metric_name cometkiwi `
    --output_dir results
```

### Fine-Tuned Model Evaluation

To evaluate a fine-tuned model, provide its checkpoint using `--checkpoint`.

```powershell
python evaluate_dataset.py `
    --data_file "PATH_TO/gate.parquet" `
    --evaluation_type gender `
    --dataset_name GATE `
    --metric_name cometkiwi `
    --checkpoint "PATH_TO/model.ckpt" `
    --output_dir results
```

If no checkpoint is provided, the pretrained model is used.

The default batch size is `32`, but it can be changed using `--batch_size`.

## 4. Evaluation Results

Each evaluation produces two files:

```text
results/
├── GATE_cometkiwi_gender_scored.parquet
└── GATE_cometkiwi_gender.json
```

**Scored Parquet**

Contains the original dataset together with the predicted model scores.

- General evaluation: `predicted_score`
- Gender evaluation: `score_masc`, `score_fem`

**JSON Summary**

Contains experiment information and aggregated evaluation statistics.

For gender evaluation, the main statistics include:

- Mean masculine and feminine scores
- Mean score difference (`score_masc - score_fem`)
- Mean absolute score difference
- Masculine and feminine preference percentages

For general evaluation, statistics include predicted score distributions and, when ground-truth scores are available, Pearson and Spearman correlations.

## 5. Generate Reports

The `generate_report.py` script creates comparison tables from the saved evaluation JSON files.

### Gender Report

```powershell
python generate_report.py `
    --results_dir results `
    --report_type gender
```

### General Report

```powershell
python generate_report.py `
    --results_dir results `
    --report_type general
```

Reports can optionally be filtered by dataset or model.

```powershell
python generate_report.py `
    --results_dir results `
    --report_type gender `
    --dataset_name GATE
```

The generated comparison tables are printed in the terminal.

This allows comparison of pretrained and fine-tuned models across the evaluation datasets.