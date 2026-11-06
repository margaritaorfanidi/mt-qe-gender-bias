
"""Merge COMET-Kiwi baseline scores across target languages for one source language."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASELINE_SCORES_DIR,
    SOURCE_LANGUAGES,
    TARGET_LANGUAGES,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_language", type=str, required=True, help="Source language")
    parser.add_argument("--output_dir", type=Path, default=BASELINE_SCORES_DIR, help="Directory for merged baseline scores")

    return parser.parse_args()


def validate_parquet(file_path):
    with open(file_path, "rb") as file:
        start_bytes = file.read(4)
        file.seek(-4, 2)
        end_bytes = file.read(4)

    return start_bytes == b"PAR1" and end_bytes == b"PAR1"


def validate_source_language(source_language):
    if source_language not in SOURCE_LANGUAGES:
        raise ValueError(
            f"Invalid source language: {source_language}. "
            f"Valid options are: {SOURCE_LANGUAGES}"
        )


def get_input_files(source_language):
    files = [
        BASELINE_SCORES_DIR /
        f"gambit_{source_language}-{target}_cometkiwi.parquet"
        for target in TARGET_LANGUAGES
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

    source_language = args.source_language.lower()
    output_dir = args.output_dir

    validate_source_language(source_language)

    files = get_input_files(source_language)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (output_dir / f"gambit_{source_language}_all_targets_cometkiwi.parquet")

    df_source = pd.concat([pd.read_parquet(file) for file in files],ignore_index=True)

    df_source.to_parquet(output_file, index=False)

    if not validate_parquet(output_file):
        raise RuntimeError("Parquet validation failed.")

    print("\n" + "=" * 60)
    print(f"{source_language} MERGE COMPLETED")
    print("=" * 60)

    print("Files merged:", len(files))
    print("Total rows:", f"{len(df_source):,}")

    print("\nLanguage pairs:")
    print(df_source["lang_pair"].value_counts().sort_index())

    print("\nSaved to:")
    print(output_file.resolve())


if __name__ == "__main__":
    main()
