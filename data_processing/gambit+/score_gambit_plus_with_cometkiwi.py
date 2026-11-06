
# This script calculates COMET-Kiwi scores for the Gambit+ dataset translations.
# per language pair

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (HF_CACHE_DIR, SOURCE_LANGUAGES, TARGET_LANGUAGES, HF_TOKEN, BASELINE_SCORES_DIR)
import argparse
import torch
from comet import download_model, load_from_checkpoint
from datasets import load_dataset


def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source_language", type=str,required=True, help="Source language")
    parser.add_argument("--target_language", type=str,required=True, help="Target language")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for scoring")    
    parser.add_argument("--output_dir", type=Path, default=BASELINE_SCORES_DIR, help="Directory for baseline scores")

    return parser.parse_args()


def get_gpu_count():
    if torch.cuda.is_available():
        print("CUDA GPU available.")
        print("Number of GPUs used:", 1)
        print("GPU 0:", torch.cuda.get_device_name(0)
        )
        return 1
    else:
        print("CUDA GPU not available. Using CPU.")
        return 0


def load_comet_kiwi():
    print("\nLoading COMET-Kiwi...")
    model_path = download_model("Unbabel/wmt22-cometkiwi-da",saving_directory=HF_CACHE_DIR)
    model = load_from_checkpoint(model_path)
    print("COMET-Kiwi loaded.")
    return model


def load_gambit_plus(language_pair):    
    print(f"\nLoading GAMBIt+ {language_pair}...")
    dataset = load_dataset(
        "ailsntua/gambit-plus",
        data_files=(f"gambit_plus.{language_pair}.jsonl"),
        split="train",
        token=HF_TOKEN,
        cache_dir=str(HF_CACHE_DIR)
    )

    df = (dataset.to_pandas().reset_index(drop=True))
    total_rows = len(df)

    print("Dataset rows:",f"{total_rows:,}")

    if total_rows == 0:
        raise RuntimeError(f"No rows found for {language_pair}")

    return df


def cometkiwi_score(df, model,gpu_count,batch_size=32):

    masc_data = [{"src": str(src),"mt": str(mt)} for src, mt in zip(df["source"], df["masc"])]
    fem_data = [{"src": str(src),"mt": str(mt)} for src, mt in zip(df["source"], df["fem"])]


    print("Original dataset rows:", f"{len(df):,}")
    print("Masculine translations:",f"{len(masc_data):,}")
    print("Feminine translations:",f"{len(fem_data):,}")

    print("Scoring...")

    masc_output = model.predict(masc_data,batch_size=batch_size,gpus=gpu_count,progress_bar=True)
    masc_scores = [float(score)for score in masc_output.scores]    

    fem_output = model.predict(fem_data,batch_size=batch_size,gpus=gpu_count,progress_bar=True)
    fem_scores = [float(score)for score in fem_output.scores]

    df["masc_score"] = masc_scores
    df["fem_score"] = fem_scores
    df["score_diff"] = (df["masc_score"]- df["fem_score"])
    df["score_mean"] = (df["masc_score"]+ df["fem_score"]) / 2
    df["abs_score_diff"] = ( df["score_diff"].abs())
   
    return df


def validate_parquet(file_path):
    with open(file_path, "rb") as file:
        start_bytes = file.read(4)
        file.seek(-4, 2)
        end_bytes = file.read(4)

    return start_bytes == b"PAR1" and end_bytes == b"PAR1"

def show_example(df):
    example = (df.head(1)).iloc[0]  
    for column in example.index:
        print(f"{column}: {example[column]}")

def validate_inputs(source_language,target_language, batch_size):

    if source_language not in SOURCE_LANGUAGES:
        raise ValueError(
            f"Invalid source language: {source_language}. "
            f"Valid options are: {SOURCE_LANGUAGES}"
        )

    if target_language not in TARGET_LANGUAGES:
        raise ValueError(
            f"Invalid target language: {target_language}. "
            f"Valid options are: {TARGET_LANGUAGES}"
        )
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")


def main():
    args = get_args()

    source_language = args.source_language.lower()
    target_language = args.target_language.lower()
    batch_size = args.batch_size

    validate_inputs(source_language, target_language, batch_size)

    language_pair = f"{source_language}-{target_language}"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    final_file = output_dir / f"gambit_{language_pair}_cometkiwi.parquet"

    gpu_count = get_gpu_count()
    
    model = load_comet_kiwi()

    df = load_gambit_plus(language_pair)
    
    df = df.head(5).copy()
    print (f"Scoring top 5 rows for language pair: {language_pair}")
    
    df = cometkiwi_score(df, model, gpu_count, batch_size=batch_size)
    
    df.to_parquet(final_file, index=False)  

    if not validate_parquet(final_file):
        raise RuntimeError("Parquet validation failed.")
     
    print("\n" + "=" * 70)
    print(f"{language_pair} FINISHED")
    print("=" * 70)
    print("Rows:",f"{len(df):,}")
    print("Saved to:", final_file.resolve())
    print("Parquet validation:", "OK")
    print("\nExample row:")
    show_example(df)

if __name__ == "__main__":
    main()