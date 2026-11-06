

import argparse
from pathlib import Path

import pandas as pd


LANGUAGES = ["ar", "de", "es", "fr", "hi", "it", "pt", "ru"]


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path where the processed parquet file will be saved."
    )

    return parser.parse_args()


def load_mt_geneval_counterfactual(languages):
    """
    Load the MT-GenEval counterfactual test sets.

    Each original example creates:
        - one feminine-source example
        - one masculine-source example
    """

    all_data = []

    for lang in languages:

        url = (
            "https://huggingface.co/datasets/gsarti/mt_geneval/"
            "resolve/refs%2Fconvert%2Fparquet/"
            f"sentences_en_{lang}/test/0000.parquet"
        )

        df = pd.read_parquet(url)

        # Feminine source
        df_female = pd.DataFrame({
            "id": df["orig_id"],
            "source": df["source_feminine"],
            "masc": df["reference_masculine"],
            "fem": df["reference_feminine"],
            "lang_pair": f"en-{lang}",
            "gender": "female",
        })

        # Masculine source
        df_male = pd.DataFrame({
            "id": df["orig_id"],
            "source": df["source_masculine"],
            "masc": df["reference_masculine"],
            "fem": df["reference_feminine"],
            "lang_pair": f"en-{lang}",
            "gender": "male",
        })

        all_data.append(df_female)
        all_data.append(df_male)

    return pd.concat(
        all_data,
        ignore_index=True
    )


def main():

    args = get_args()

    data = load_mt_geneval_counterfactual(LANGUAGES)

    print("\nMT-GenEval counterfactual dataset")
    print("Shape:", data.shape)
    print("Columns:", data.columns.tolist())

    print("\nLanguage pairs:")
    print(data["lang_pair"].value_counts())

    print("\nGender distribution:")
    print(data["gender"].value_counts())

    print("\nSample:")
    print(data.head().to_string(index=False))

    args.output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_parquet(
        args.output_file,
        index=False
    )

    print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()
