
"""Build the GATE evaluation dataset using all 2-variant examples."""

import argparse
import subprocess
from pathlib import Path

import pandas as pd


REPO_URL = "https://github.com/MicrosoftTranslator/GATE.git"

LANGUAGES = {
    "ES": "es",
    "FR": "fr",
    "IT": "it",
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo_dir",
        type=Path,
        required=True,
        help="Directory for the GATE repository."
    )

    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path for the processed Parquet file."
    )

    return parser.parse_args()


def clone_repository(repo_dir):
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)

        print("Cloning GATE repository...")

        subprocess.run(
            ["git", "clone", REPO_URL, str(repo_dir)],
            check=True
        )
    else:
        print("GATE repository already exists.")


def load_gate(repo_dir):
    all_data = []

    for folder, lang in LANGUAGES.items():

        file = (
            repo_dir
            / "data"
            / folder
            / f"{folder}_2_variants.tsv"
        )

        print(f"Reading: {file}")

        if not file.is_file():
            raise FileNotFoundError(f"File not found: {file}")

        df = pd.read_csv(file, sep="\t")

        required_columns = ["source", "m", "f"]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in {file}: {missing}"
            )

        df_out = pd.DataFrame({
            "id": [f"en-{lang}_{i}" for i in range(len(df))],
            "source": df["source"],
            "masc": df["m"],
            "fem": df["f"],
            "lang_pair": f"en-{lang}",
        })

        all_data.append(df_out)

    return pd.concat(all_data, ignore_index=True)


def main():
    args = get_args()

    clone_repository(args.repo_dir)

    data = load_gate(args.repo_dir)

    print("\nGATE dataset")
    print("Shape:", data.shape)
    print("Columns:", data.columns.tolist())

    print("\nLanguage pairs:")
    print(data["lang_pair"].value_counts())

    print("\nSample:")
    print(data.head().to_string(index=False))

    args.output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_parquet(args.output_file, index=False)

    print(f"\nSaved to: {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
