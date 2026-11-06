
# this script is used for eda the Gambit+ dataset 


from huggingface_hub.utils import disable_progress_bars
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import (HF_CACHE_DIR,SOURCE_LANGUAGES,TARGET_LANGUAGES,HF_TOKEN)





# def id_diagnostics(df):
#     print("\n" + "=" * 70)
#     print("Key Diagnostics")
#     print("=" * 70)
#     if  df["id"].is_unique:
#         print("IDs are unique.")
#     else:
#         print("The language pair IDs are not unique.")
#     if not df.duplicated(subset=["id", "lang_pair"]).any():
#         print("All (ID, lang_pair) combinations are unique.")
#     else:
#         print("There are duplicate (ID, lang_pair) combinations.")


def dataset_statistics(df):
    print("\n" + "=" * 70)
    print("GAMBIt+ Dataset Information")
    print("=" * 70)
    print(f"Total rows:           {len(df):,}")
    print(f"Columns:              {len(df.columns)}")
    print(f"Unique rows:          {len(df.drop_duplicates()):,}")
    print(f"Language pairs:       {df['lang_pair'].nunique()}")
    print(f"Unique source texts:  {df['source'].nunique():,}")
    print(f"Unique masc texts:    {df['masc'].nunique():,}")
    print(f"Unique fem texts:     {df['fem'].nunique():,}")
    print(f"Missing values:        {df.isna().sum().sum():,}")
    print(f"Identical masculine/feminine translations: {((df['masc'] == df['fem']).sum()):,}")

def text_length_statistics(df, threshold=1.0):
    print("\n" + "=" * 70)
    print("Mean word counts: source / masculine / feminine")
    print("=" * 70)
    source_mean = df["source"].str.split().str.len().mean()
    masc_mean   = df["masc"].str.split().str.len().mean()
    fem_mean    = df["fem"].str.split().str.len().mean()

    print(f"Source mean words:     {source_mean:.2f}")
    print(f"Masculine mean words:  {masc_mean:.2f}")
    print(f"Feminine mean words:   {fem_mean:.2f}")

    diff = masc_mean - fem_mean

    print(f"\nMasc - Fem mean difference: {diff:.2f}")

    if abs(diff) > threshold:
        longer = "masculine" if diff > 0 else "feminine"
        print(f"Large difference — {longer} translations are longer on average.")
    else:
        print("Difference within threshold — roughly balanced.")

def source_target_distribution(df):
    print("\n" + "=" * 70)
    print("Source x Target Distribution")
    print("=" * 70)

    pair_table = (
        df.assign(
            source_lang=df["lang_pair"].str.split("-").str[0],
            target_lang=df["lang_pair"].str.split("-").str[1],
        ).pivot_table(
            index="source_lang",
            columns="target_lang",
            values="id",
            aggfunc="count",
            fill_value=0,
        )
    )

    print(pair_table.to_string())
    return pair_table


def print_example(df, random_state=42):
    print("\n" + "=" * 70)
    print("GAMBIt+ Example")
    print("=" * 70)

    example = df.sample(n=1, random_state=random_state).iloc[0]

    for column in df.columns:
        print(f"\n{column.upper()}:")
        print(example[column])


def load_gambit_dataset():
    print("\nLoading GAMBIt+...")

    # Hide the many download / resolving progress bars.
    disable_progress_bar()
    disable_progress_bars()

    # GAMBIt+ data files
    data_files = [
        f"gambit_plus.{src}-{tgt}.jsonl"
        for src in SOURCE_LANGUAGES
        for tgt in TARGET_LANGUAGES
    ]
    # Load the dataset from Hugging Face using the specified data files
    dataset = load_dataset(
        "ailsntua/gambit-plus",
        data_files=data_files,
        split="train",
        token=HF_TOKEN,
        cache_dir=str(HF_CACHE_DIR),
    )

    df = (dataset.to_pandas().reset_index(drop=True))
    return df


def isco_distribution(df):
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

    print("\n" + "=" * 70)
    print("ISCO-08 Major Group Distribution")
    print("=" * 70)

    isco_codes = (df["meta_domain"].astype(str).str.zfill(4))

    major_codes = (isco_codes.str[0])

    major_names = (major_codes.map(ISCO_MAJOR_GROUPS))

    distribution = (major_names.value_counts().rename_axis("major_group").reset_index(name="samples"))

    distribution["percentage"] = (distribution["samples"] / len(df)* 100)

    print(
        distribution.to_string(
            index=False,
            formatters={
                "samples": lambda x: f"{x:,}",
                "percentage": lambda x: f"{x:.2f}%",
            }
        )
    )

    return distribution

def main():
    
    df = load_gambit_dataset()
    df.info()
    print("\n" + "=" * 70)

    dataset_statistics(df)
    source_target_distribution(df)
    # checks what is the key of the table 
    # id_diagnostics(df)
    text_length_statistics(df)
    # isco distribution must be checked 
    # isco_distribution(df)
    print_example(df)


if __name__ == "__main__":
    main()