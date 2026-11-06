# Evaluation Dataset Preparation

This folder contains scripts for downloading, processing, and converting datasets into standardized Parquet files used to evaluate fine-tuned Machine Translation Quality Estimation (QE) models.

We use **three categories of evaluation datasets**, each designed to assess a different aspect of model performance.

## 1. Gender-Ambiguous Evaluation

**Datasets:**
- GATE
- MT-GenEval Context

These datasets contain gender-ambiguous source sentences paired with masculine and feminine translation variants. They are used to investigate whether QE models assign different quality scores to translations that differ in grammatical gender.

**Required Parquet columns:**

| Column | Description |
|---|---|
| `source` | Original source sentence |
| `masc` | Masculine translation |
| `fem` | Feminine translation |
| `lang_pair` | Source-target language pair (e.g., `en-es`) |

An additional `id` column may be included to identify individual examples.

## 2. Gender-Unambiguous Evaluation

**Dataset:**
- MT-GenEval Counterfactual

This dataset contains source sentences with explicit gender information. Each original example is transformed into two examples, one with a masculine source and one with a feminine source.

It is used to evaluate how QE models respond to gender information explicitly expressed in the source sentence.

**Required Parquet columns:**

| Column | Description |
|---|---|
| `id` | Identifier linking counterfactual examples |
| `source` | Masculine or feminine source sentence |
| `masc` | Masculine translation |
| `fem` | Feminine translation |
| `lang_pair` | Source-target language pair |
| `gender` | Gender expressed in the source (`male` or `female`) |

## 3. General Quality Estimation Evaluation

**Dataset:**
- MLQE-PE

This dataset contains source sentences, machine translations, and human quality assessment scores.

Unlike the gender-focused datasets, it is used to assess the overall quality estimation performance of the models and determine whether fine-tuning affects their ability to predict translation quality.

**Required Parquet columns:**

| Column | Description |
|---|---|
| `id` | Unique example identifier |
| `source` | Original source sentence |
| `mt` | Machine-generated translation |
| `human_score` | Human quality assessment score (`z_mean`) |
| `lang_pair` | Source-target language pair |

## Output Format

All scripts save the processed datasets in **Parquet format (`.parquet`)** under:

```text
data/
└── evaluation_data/
    ├── gate_all.parquet
    ├── mt_geneval_context.parquet
    ├── mt_geneval_counterfactual.parquet
    └── mlqe_pe_test.parquet
```

The gender-focused datasets follow the common `source`, `masc`, `fem`, `lang_pair` schema, with additional columns where necessary.

The general QE dataset follows the `source`, `mt`, `human_score`, `lang_pair` schema.

All scripts are responsible only for dataset preparation and standardization. Model scoring and evaluation are performed separately.