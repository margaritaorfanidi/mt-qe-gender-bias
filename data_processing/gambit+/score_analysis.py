
from pathlib import Path
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=Path, required=True, help="Path to the scored parquet file")

    return parser.parse_args()



def general_info(df):

    df["source_lang"] = (df["lang_pair"].str.split("-").str[0])

    df["target_lang"] = (df["lang_pair"].str.split("-").str[1])

    print("\n" + "=" * 70)
    print("General File Statistics")
    print("=" * 70)
    print("Total rows:", len(df))
    print("Columns:", len(df.columns))
    print("Language pairs:", df["lang_pair"].nunique())
    print("Source languages:", df["source_lang"].nunique())
    print("Target languages:", df["target_lang"].nunique())
    print("Duplicate rows:", df.duplicated().sum())


def source_target_distr(df):
    print("\n" + "=" * 70)
    print("Source x Target Distribution")
    print("=" * 70)

    pair_table = (df.pivot_table(index="source_lang",columns="target_lang", values="id",aggfunc="count",fill_value=0))

    print( pair_table.to_string() )


def general_score_statistics(df):
    print("\n" + "=" * 70)
    print("Dataset statistics ")
    print("=" * 70)

    score_columns = ["masc_score","fem_score","score_diff","abs_score_diff","score_mean" ]

    print(df[score_columns].describe().round(4).to_string())

def score_direction(df):

    print("\n" + "=" * 70)
    print("Score direction")
    print("=" * 70)

    masc_higher = (df["score_diff"] > 0).sum()
    fem_higher = (df["score_diff"] < 0).sum()
    equal_scores = (df["score_diff"] == 0).sum()

    print("Masculine scored higher:", masc_higher,f"({masc_higher / len(df) * 100:.2f}%)")

    print("Feminine scored higher:", fem_higher,f"({fem_higher / len(df) * 100:.2f}%)")

    print("Equal scores:", equal_scores,f"({equal_scores / len(df) * 100:.2f}%)")


    print("\n" + "=" * 70)
    print("Score Direction per language pair")
    print("=" * 70)

    direction_stats = (df.assign(
            masc_higher=df["score_diff"] > 0,
            fem_higher=df["score_diff"] < 0,
            equal=df["score_diff"] == 0
        )
        .groupby("lang_pair")
        .agg(
            rows=("id", "size"),
            masc_higher_pct=("masc_higher", "mean"),
            fem_higher_pct=("fem_higher", "mean"),
            equal_pct=("equal", "mean")
        )
    )

    direction_stats[["masc_higher_pct",  "fem_higher_pct", "equal_pct" ] ] *= 100

    print(direction_stats.round(2).to_string())


def isco_analysis(df):

    ISCO_MAJOR_GROUPS = {
        "0": "Armed Forces Occupations",
        "1": "Managers",
        "2": "Professionals",
        "3": "Technicians and Associate Professionals",
        "4": "Clerical Support Workers",
        "5": "Service and Sales Workers",
        "6": "Skilled Agricultural, Forestry and Fishery Workers",
        "7": "Craft and Related Trades Workers",
        "8": "Plant and Machine Operators and Assemblers",
        "9": "Elementary Occupations",
    }

    df["isco_major_code"] = (df["meta_domain"].astype(str).str.zfill(4).str[0])

    df["isco_major_group"] = (df["isco_major_code"].map(ISCO_MAJOR_GROUPS))

    print("\n" + "=" * 70)
    print("SCORE STATISTICS PER ISCO-08 MAJOR GROUP")
    print("=" * 70)

    isco_stats = (
        df
        .groupby("isco_major_group")
        .agg(
            rows=("id", "size"),
            masc_mean=("masc_score", "mean"),
            fem_mean=("fem_score", "mean"),
            mean_diff=("score_diff", "mean"),
            mean_abs_diff=("abs_score_diff", "mean")
        )
        .sort_values(
            "mean_diff",
            ascending=False
        )
        .round(4)
    )

    print(
        isco_stats.to_string()
    )

def source_score_statistics(df):
    print("\n" + "=" * 70)
    print("SCORE STATISTICS PER SOURCE LANGUAGE")
    print("=" * 70)

    source_stats = (
        df
        .groupby("source_lang")
        .agg(
            rows=("id", "size"),
            masc_mean=("masc_score", "mean"),
            fem_mean=("fem_score", "mean"),
            mean_diff=("score_diff", "mean"),
            mean_abs_diff=("abs_score_diff", "mean")
        )
        .round(4)
    )

    print(
        source_stats.to_string()
    )

def target_score_statistics(df):
    print("\n" + "=" * 70)
    print("Score Statistics Per Target Language")
    print("=" * 70)

    target_stats = (
        df
        .groupby("target_lang")
        .agg(
            rows=("id", "size"),
            masc_mean=("masc_score", "mean"),
            fem_mean=("fem_score", "mean"),
            mean_diff=("score_diff", "mean"),
            mean_abs_diff=("abs_score_diff", "mean")
        )
        .round(4)
    )

    print(
        target_stats.to_string()
    )


def lang_pair_score_statistics(df):
    print("\n" + "=" * 70)
    print("Score Statistics Per Language Pair")
    print("=" * 70)

    pair_stats = (
        df
        .groupby("lang_pair")
        .agg(
            rows=("id", "size"),
            masc_mean=("masc_score", "mean"),
            fem_mean=("fem_score", "mean"),
            mean_diff=("score_diff", "mean"),
            mean_abs_diff=("abs_score_diff", "mean")
        )
        .round(4)
    )

    print(
        pair_stats.to_string()
    )


def largest_score_difference(df):
    print("\n" + "=" * 70)
    print("Largest Absolute Score Differences")
    print("=" * 70)

    largest_differences = (
        df
        .nlargest(
            10,
            "abs_score_diff"
        )
        [
            [
                "id",
                "lang_pair",
                "masc_score",
                "fem_score",
                "score_diff",
                "abs_score_diff"
            ]
        ]
    )

    print(
        largest_differences
        .round(4)
        .to_string(index=False)
    )

def main():

    args = get_args()

    input_file = args.input_file

    if not input_file.exists():
        raise FileNotFoundError(
            f"File not found:\n{input_file}"
        )

    df = pd.read_parquet(input_file)

    general_info(df)

    source_target_distr(df)

    general_score_statistics(df)

    score_direction(df)

    if "meta_domain" in df.columns:
        isco_analysis(df)

    source_score_statistics(df)

    target_score_statistics(df)

    lang_pair_score_statistics(df)

    largest_score_difference(df)


if __name__ == "__main__":
    main()