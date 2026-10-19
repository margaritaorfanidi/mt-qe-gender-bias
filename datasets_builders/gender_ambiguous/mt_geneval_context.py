import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LANGUAGES = ["ar", "de", "es", "fr", "hi", "it", "pt", "ru"]

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file",type=Path, required=True, help="Path where the processed parquet file will be saved.")

    return parser.parse_args()

def load_mt_geneval(languages):
    """
    Load the MT-GenEval context test sets.

    Each example contains:
        - context: full context containing gender information
        - source: ambiguous sentence
        - reference_original: translation matching the context gender
        - reference_flipped: translation with the opposite gender
    """

    all_data = []

    for lang in languages:

        url = ("https://huggingface.co/datasets/gsarti/mt_geneval/"
            f"resolve/refs%2Fconvert%2Fparquet/"
            f"context_en_{lang}/test/0000.parquet"
        )

        df = pd.read_parquet(url)

        df["lang_pair"] = f"en-{lang}"

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def load_gender_annotations(languages):
    """
    Load gender annotations for MT-GenEval context examples.
    """

    all_genders = []

    for lang in languages:

        url = (
            "https://raw.githubusercontent.com/"
            "deep-spin/gender-bias-qe-metrics/main/"
            "data/mtgeneval/context_genders/"
            f"geneval-context-en{lang}-genders.txt"
        )

        gender_df = pd.read_csv(url, sep="\t")

        gender_df["lang_pair"] = f"en-{lang}"

        all_genders.append(gender_df)

    return pd.concat(all_genders, ignore_index=True)


def prepare_evaluation_data(data, genders):
    """
    Create explicit masculine and feminine translations.
    """

    data = data.copy()

    data["gender"] = (genders["gender"].reset_index(drop=True))

    data["masc"] = np.where(data["gender"] == "male", data["reference_original"],data["reference_flipped"])

    data["fem"] = np.where(data["gender"] == "female",data["reference_original"],data["reference_flipped"])

    return data[["source","masc","fem","lang_pair" ]]


def validate_data(data, genders):

    if len(data) != len(genders):
        return False

    same_sources = (
        data["source"].reset_index(drop=True)
        == genders["source"].reset_index(drop=True)
    ).all()

    same_lang_pairs = (
        data["lang_pair"].reset_index(drop=True)
        == genders["lang_pair"].reset_index(drop=True)
    ).all()

    return same_sources and same_lang_pairs

def main():
    args = get_args()

    data = load_mt_geneval(LANGUAGES)

   
    genders = load_gender_annotations(LANGUAGES)
    if not validate_data(data, genders):
        raise ValueError("Data and gender annotations do not match.")

    evaluation_data = prepare_evaluation_data(data, genders)

    print("\nFinal dataset")
    print("Shape:", evaluation_data.shape)
    print("Columns:", evaluation_data.columns.tolist())

    print("\nSample:")
    print(evaluation_data.head().to_string(index=False))

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    evaluation_data.to_parquet(args.output_file, index=False)

    print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()