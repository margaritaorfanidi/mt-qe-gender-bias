import argparse
from pathlib import Path
import pandas as pd

keywords = ["they", "them"]
CHUNK_SIZE = 100


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True, help="Parquet file to process.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    return parser.parse_args()


def has_key_words(series, keywords):
    mask = pd.Series(False, index=series.index)
    for k in keywords:
        mask = mask | series.str.contains(k, case=False, na=False, regex=False)
    return mask


def main():
    args = get_args()

    data = pd.read_parquet(args.file)
    data["neutral_gender"] = has_key_words(data["source"], keywords)

    unique_sources = data.drop_duplicates(subset="source")
    count = unique_sources["neutral_gender"].sum()
    print(f"Βρέθηκαν {count} από {len(unique_sources)} μοναδικές προτάσεις ({count / len(unique_sources):.2%})")

    matches = unique_sources.loc[unique_sources["neutral_gender"], ["source"]].reset_index(drop=True)

    if matches.empty:
        print("Δεν βρέθηκαν προτάσεις, δεν σώθηκε τίποτα.")
        return

    out_path = args.output or args.file.with_name(args.file.stem + "_neutral.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_files = 0
    for start in range(0, len(matches), CHUNK_SIZE):
        n_files += 1
        chunk = matches.iloc[start:start + CHUNK_SIZE]
        chunk_path = out_path.with_name(f"{out_path.stem}_{n_files:03d}.csv")
        chunk.to_csv(chunk_path, index=False, encoding="utf-8-sig")

    print(f"Saved {n_files} files ({CHUNK_SIZE} rows each) to {out_path.parent}")


if __name__ == "__main__":
    main()