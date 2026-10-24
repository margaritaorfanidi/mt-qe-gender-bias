
import argparse
import csv
import subprocess
import tarfile
from pathlib import Path

import pandas as pd


REPO_URL = "https://github.com/sheffieldnlp/mlqe-pe.git"


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo_dir",
        type=Path,
        required=True,
        help="Path where the MLQE-PE repository will be cloned."
    )

    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path where the processed parquet file will be saved."
    )

    return parser.parse_args()


def clone_repository(repo_dir):
    """
    Clone the MLQE-PE repository if it does not exist.
    """

    if not repo_dir.exists():

        repo_dir.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        subprocess.run(
            ["git", "clone", REPO_URL, str(repo_dir)],
            check=True
        )

    else:
        print("MLQE-PE repository already exists.")


def extract_archives(data_dir):
    """
    Extract the MLQE-PE test archives.
    """

    for archive in data_dir.rglob("*.tar.gz"):

        print(f"Extracting: {archive.name}")

        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(
                path=archive.parent,
                filter="data"
            )


def load_mlqe_pe(data_dir):
    """
    Load all MLQE-PE Direct Assessment test files.
    """

    all_data = []

    for file in sorted(data_dir.rglob("*.tsv")):

        print(f"Reading: {file}")

        df = pd.read_csv(
            file,
            sep="\t",
            quoting=csv.QUOTE_NONE,
            engine="python"
        )

        lang_pair = file.parent.name

        df["lang_pair"] = lang_pair

        all_data.append(df)

    if not all_data:
        raise FileNotFoundError(
            f"No TSV files found in {data_dir}"
        )

    return pd.concat(
        all_data,
        ignore_index=True
    )


def prepare_evaluation_data(data):
    """
    Convert MLQE-PE into the general QE evaluation format.
    """

    data = data.copy()

    required_columns = [
        "original",
        "translation",
        "z_mean",
        "lang_pair"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data["id"] = (
        data["lang_pair"]
        + "_"
        + data.groupby("lang_pair").cumcount().astype(str)
    )

    return data.rename(
        columns={
            "original": "source",
            "translation": "mt",
            "z_mean": "human_score"
        }
    )[
        ["id", "source", "mt", "human_score", "lang_pair"]
    ]


def main():

    args = get_args()

    clone_repository(args.repo_dir)

    data_dir = (
        args.repo_dir
        / "data"
        / "direct-assessments"
        / "test"
    )

    extract_archives(data_dir)

    data = load_mlqe_pe(data_dir)

    evaluation_data = prepare_evaluation_data(data)

    print("\nFinal dataset")
    print("Shape:", evaluation_data.shape)
    print("Columns:", evaluation_data.columns.tolist())

    print("\nLanguage pairs:")
    print(evaluation_data["lang_pair"].value_counts())

    print("\nSample:")
    print(evaluation_data.head().to_string(index=False))

    args.output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation_data.to_parquet(
        args.output_file,
        index=False
    )

    print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()
