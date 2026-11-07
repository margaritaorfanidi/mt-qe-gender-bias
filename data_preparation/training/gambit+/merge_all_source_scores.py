
"""Merge COMET-Kiwi baseline scores from all source languages."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import BASELINE_SCORES_DIR, SOURCE_LANGUAGES


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=Path, default=BASELINE_SCORES_DIR, help="Directory for merged baseline scores")

    return parser.parse_args()


def validate_parquet(file_path):
    with open(file_path, "rb") as file:
        start_bytes = file.read(4)
        file.seek(-4, 2)
        end_bytes = file.read(4)

    return start_bytes == b"PAR1" and end_bytes == b"PAR1"


def get_input_files():
    files = [
        BASELINE_SCORES_DIR /
        f"gambit_{source}_all_targets_cometkiwi.parquet"
        for source in SOURCE_LANGUAGES
    ]

    missing_files = [
        file for file in files if not file.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing score files:\n" +
            "\n".join(str(file) for file in missing_files)
        )

    return files


def main():
    args = get_args()

    input_files = get_input_files()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir /
        "gambit_all_sources_all_targets_cometkiwi.parquet"
    )

    merged_df = pd.concat(
        [pd.read_parquet(file) for file in input_files],
        ignore_index=True
    )

    merged_df.to_parquet(output_file, index=False)

    if not validate_parquet(output_file):
        raise RuntimeError("Parquet validation failed.")

    print("\n" + "=" * 60)
    print("ALL SOURCES MERGE COMPLETED")
    print("=" * 60)

    print("Files merged:", len(input_files))
    print("Total rows:", f"{len(merged_df):,}")

    print("\nLanguage pairs:")
    print(merged_df["lang_pair"].value_counts().sort_index())

    print("\nSaved to:")
    print(output_file.resolve())


if __name__ == "__main__":
    main()
