
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare training data for COMET-Kiwi."
    )

    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_dir", type=Path)

    return parser.parse_args()


def split(df, holdout_size=0.2):
    """Split data into train, validation and test using grouped IDs."""

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_size,
        random_state=42
    )

    train_idx, temp_idx = next(
        splitter.split(df, groups=df["id"])
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=42
    )

    validation_idx, test_idx = next(
        splitter.split(df_temp, groups=df_temp["id"])
    )

    df_validation = df_temp.iloc[validation_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    # Check for ID leakage
    train_ids = set(df_train["id"])
    validation_ids = set(df_validation["id"])
    test_ids = set(df_test["id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    return df_train, df_validation, df_test


def split_gender_pairs(data):
    """Convert masculine and feminine translations into separate rows."""

    masc = data.copy()
    masc["mt"] = masc["masc"]
    masc["gender"] = "masc"
    masc["score"] = masc["score_mean"]

    fem = data.copy()
    fem["mt"] = fem["fem"]
    fem["gender"] = "fem"
    fem["score"] = fem["score_mean"]

    return pd.concat([masc, fem], ignore_index=True)


def prepare_comet_data(df):
    """Keep only the columns required for COMET-Kiwi training."""

    return (
        df.rename(columns={"source": "src"})
        [["src", "mt", "score"]]
        .copy()
    )


def print_statistics(df, splits):
    """Display dataset sizes and language pair distributions."""

    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    print(f"Total rows: {len(df):,}")
    print(f"Unique sources: {df['source'].nunique():,}")

    print("\nSplit sizes:")

    for name, data in splits.items():
        percentage = len(data) / len(df) * 100
        print(f"{name:12} {len(data):>10,} ({percentage:.2f}%)")

    distribution = pd.DataFrame({
        name: data["lang_pair"].value_counts()
        for name, data in splits.items()
    }).fillna(0).astype(int)

    print("\nLanguage pair distribution:")
    print(distribution.sort_index())

    print("\nID leakage check passed.")


def print_example(df, lang_pair="en-el"):
    """Display one masculine/feminine example from the training set."""

    examples = df[df["lang_pair"] == lang_pair]

    if examples.empty:
        print(f"\nNo examples found for {lang_pair}.")
        return

    example = examples.iloc[0]

    print("\n" + "=" * 50)
    print(f"EXAMPLE: {lang_pair.upper()}")
    print("=" * 50)

    print(f"\nID: {example['id']}")
    print(f"Source: {example['source']}")
    print(f"Score: {example['score_mean']}")

    print(f"\nMasculine: {example['masc']}")
    print(f"Feminine:  {example['fem']}")


def main():
    args = get_args()

    input_file = args.input_file
    output_dir = args.output_dir

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_parquet(input_file)

    # Create train, validation and test splits
    df_train, df_validation, df_test = split(df)

    splits = {
        "train": df_train,
        "validation": df_validation,
        "test": df_test
    }

    # Display statistics and an example
    print_statistics(df, splits)
    print_example(df_train)

    # Save wide-format Parquet and COMET-compatible CSV files
    for name, data in splits.items():

        data.to_parquet(
            output_dir / f"{name}.parquet",
            index=False
        )

        long_data = split_gender_pairs(data)
        assert len(long_data) == 2 * len(data)

        comet_data = prepare_comet_data(long_data)

        comet_data.to_csv(
            output_dir / f"{name}_comet.csv",
            index=False
        )

        print(
            f"Saved {name}: {len(data):,} original rows, "
            f"{len(comet_data):,} COMET samples"
        )

    print(f"\nAll files saved to: {output_dir}")


if __name__ == "__main__":
    main()
