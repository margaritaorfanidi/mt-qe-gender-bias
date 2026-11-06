
from pathlib import Path
import os


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# Data
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "train_data"

BASELINE_SCORES_DIR = DATA_DIR / "baseline_scores"

DATA_SPLITS_DIR = DATA_DIR / "data_splits"

FINETUNED_SCORES_DIR = DATA_DIR / "finetuned_scores"


# ============================================================
# Models and logs
# ============================================================

MODELS_DIR = PROJECT_ROOT / "models"

LOGS_DIR = PROJECT_ROOT / "logs"


# ============================================================
# Hugging Face
# ============================================================

HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# Languages
# ============================================================

SOURCE_LANGUAGES = (
    "en",
    "fi",
    "tr",
)

TARGET_LANGUAGES = (
    "ar",
    "cs",
    "el",
    "es",
    "fr",
    "is",
    "it",
    "pt",
    "ru",
    "sr",
    "uk",
)
