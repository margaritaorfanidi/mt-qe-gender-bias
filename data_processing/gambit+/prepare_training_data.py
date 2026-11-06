# Prepares train, validation and test splits for COMET-Kiwi training.


# ============================================================
# Libraries
# ============================================================

import sys
from pathlib import Path

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# Arguments
# ============================================================

if len(sys.argv) != 3:
    raise ValueError(
        "Give input parquet file and output directory.\n"
        "Example:\n"
        "python -m data_preparation.prepare_training_data "
        "/kaggle/working/baseline_scores/"
        "gambit_all_sources_all_targets_cometkiwi.parquet "
        "/kaggle/working/data_splits"
    )

input_file = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

if not input_file.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{input_file}"
    )

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Output files
# ============================================================

train_file = (
    output_dir
    / "train.parquet"
)

validation_file = (
    output_dir
    / "validation.parquet"
)

test_file = (
    output_dir
    / "test.parquet"
)

train_comet_file = (
    output_dir
    / "train_comet.csv"
)

validation_comet_file = (
    output_dir
    / "validation_comet.csv"
)


# ============================================================
# Load data
# ============================================================

print("\n" + "=" * 60)
print("INPUT FILE")
print("=" * 60)

print(input_file)

df = pd.read_parquet(
    input_file
)

print(
    "\nRows:",
    f"{len(df):,}"
)

print(
    "Unique sources:",
    f"{df['source'].nunique():,}"
)


# ============================================================
# Train / validation / test split
# ============================================================

# First split:
# 80% train
# 20% temporary (validation + test)
#
# Group by ID so that all rows belonging to the same
# GAMBIt+ example stay in the same split.

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, temp_idx = next(
    splitter.split(
        df,
        groups=df["id"]
    )
)

df_train = (
    df
    .iloc[train_idx]
    .reset_index(drop=True)
)

df_temp = (
    df
    .iloc[temp_idx]
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Second split:
# temporary 20% ->
# 10% validation
# 10% test
# ------------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

validation_idx, test_idx = next(
    splitter.split(
        df_temp,
        groups=df_temp["id"]
    )
)

df_validation = (
    df_temp
    .iloc[validation_idx]
    .reset_index(drop=True)
)

df_test = (
    df_temp
    .iloc[test_idx]
    .reset_index(drop=True)
)


# ============================================================
# Split statistics
# ============================================================

print("\n" + "=" * 60)
print("SPLIT STATISTICS")
print("=" * 60)

print(
    "FULL:",
    f"{len(df):,}"
)

print(
    "TRAIN:",
    f"{len(df_train):,}",
    f"({len(df_train) / len(df):.2%})"
)

print(
    "VALIDATION:",
    f"{len(df_validation):,}",
    f"({len(df_validation) / len(df):.2%})"
)

print(
    "TEST:",
    f"{len(df_test):,}",
    f"({len(df_test) / len(df):.2%})"
)


# ============================================================
# Check ID leakage
# ============================================================

train_ids = set(
    df_train["id"]
)

validation_ids = set(
    df_validation["id"]
)

test_ids = set(
    df_test["id"]
)

assert train_ids.isdisjoint(
    validation_ids
)

assert train_ids.isdisjoint(
    test_ids
)

assert validation_ids.isdisjoint(
    test_ids
)

print(
    "\nID leakage check passed."
)


# ============================================================
# Language pair distribution
# ============================================================

print("\n" + "=" * 60)
print("LANGUAGE PAIR DISTRIBUTION")
print("=" * 60)

print("\nTRAIN:")

print(
    df_train["lang_pair"]
    .value_counts()
    .sort_index()
)

print("\nVALIDATION:")

print(
    df_validation["lang_pair"]
    .value_counts()
    .sort_index()
)

print("\nTEST:")

print(
    df_test["lang_pair"]
    .value_counts()
    .sort_index()
)


# ============================================================
# Convert masculine / feminine translations to separate rows
# ============================================================

def to_long(data):

    masc = data.copy()

    masc["mt"] = (
        masc["masc"]
    )

    masc["gender"] = (
        "masc"
    )

    masc["score"] = (
        masc["score_mean"]
    )


    fem = data.copy()

    fem["mt"] = (
        fem["fem"]
    )

    fem["gender"] = (
        "fem"
    )

    fem["score"] = (
        fem["score_mean"]
    )


    long_data = pd.concat(
        [
            masc,
            fem
        ],
        ignore_index=True
    )

    return long_data


train = to_long(
    df_train
)

validation = to_long(
    df_validation
)

test = to_long(
    df_test
)


# ============================================================
# Simple check
# ============================================================

assert len(train) == (
    2 * len(df_train)
)

assert len(validation) == (
    2 * len(df_validation)
)

assert len(test) == (
    2 * len(df_test)
)

print(
    "\nMasc/Fem conversion check passed."
)

print(
    "\nRows after conversion:"
)

print(
    "TRAIN:",
    f"{len(train):,}"
)

print(
    "VALIDATION:",
    f"{len(validation):,}"
)

print(
    "TEST:",
    f"{len(test):,}"
)


# ============================================================
# Save data splits
# ============================================================

df_train.to_parquet(
    train_file,
    index=False
)

df_validation.to_parquet(
    validation_file,
    index=False
)

df_test.to_parquet(
    test_file,
    index=False
)


# ============================================================
# Prepare COMET data
# ============================================================

train_comet = (
    train
    .rename(
        columns={
            "source": "src"
        }
    )
    [
        [
            "src",
            "mt",
            "score"
        ]
    ]
    .copy()
)

validation_comet = (
    validation
    .rename(
        columns={
            "source": "src"
        }
    )
    [
        [
            "src",
            "mt",
            "score"
        ]
    ]
    .copy()
)


# ============================================================
# Save COMET datasets
# ============================================================

train_comet.to_csv(
    train_comet_file,
    index=False
)

validation_comet.to_csv(
    validation_comet_file,
    index=False
)


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 60)
print("SAVED FILES")
print("=" * 60)

print(
    train_file
)

print(
    validation_file
)

print(
    test_file
)

print(
    train_comet_file
)

print(
    validation_comet_file
)

print(
    "\nCOMET train shape:",
    train_comet.shape
)

print(
    "COMET validation shape:",
    validation_comet.shape
)


# ============================================================
# COMET train example: EN-EL masculine / feminine
# ============================================================

print("\n" + "=" * 60)
print("COMET TRAIN EXAMPLE: EN-EL MASC / FEM")
print("=" * 60)

example = train[
    (train["lang_pair"] == "en-el")
    & (train["gender"] == "masc")
].iloc[0]

same_id = example["id"]

example_indices = train[
    (train["id"] == same_id)
    & (train["lang_pair"] == "en-el")
].index


for idx in example_indices:

    gender = train.loc[
        idx,
        "gender"
    ]

    row = train_comet.loc[
        idx
    ]

    print("\n" + "-" * 60)

    print(
        "GENDER:",
        gender
    )

    print("-" * 60)

    print(
        "src:",
        row["src"]
    )

    print(
        "mt:",
        row["mt"]
    )

    print(
        "score:",
        row["score"]
    )