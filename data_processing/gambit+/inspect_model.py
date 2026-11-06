"""
Inspect the pretrained COMET-Kiwi architecture.

The goal is to understand:
1. What type of model COMET-Kiwi is.
2. Which main modules it contains.
3. What the full architecture looks like.
4. Which parameters are currently trainable.
"""

import os

# Disable Hugging Face progress bars.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub.utils import disable_progress_bars

from comet import (
    download_model,
    load_from_checkpoint,
)

from config import HF_CACHE_DIR



disable_progress_bars()


# ============================================================
# Load pretrained COMET-Kiwi
# ============================================================

print("\nLoading COMET-Kiwi...")

model_path = download_model(
    "Unbabel/wmt22-cometkiwi-da",
    saving_directory=HF_CACHE_DIR,
)

model = load_from_checkpoint(
    model_path
)

print("COMET-Kiwi loaded.")


# ============================================================
# Model type
# ============================================================

print("\n" + "=" * 70)
print("MODEL TYPE")
print("=" * 70)

print(type(model).__name__)


# ============================================================
# Main components
# ============================================================

print("\n" + "=" * 70)
print("MAIN MODEL COMPONENTS")
print("=" * 70)

for name, module in model.named_children():

    print(
        f"{name:<25} -> {type(module).__name__}"
    )


# ============================================================
# Full architecture
# ============================================================

print("\n" + "=" * 70)
print("FULL MODEL ARCHITECTURE")
print("=" * 70)

print(model)


# ============================================================
# Important COMET components
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT COMPONENTS")
print("=" * 70)

if hasattr(model, "encoder"):
    print(
        f"Encoder:              "
        f"{type(model.encoder).__name__}"
    )

if hasattr(model, "layerwise_attention"):
    print(
        f"Layerwise attention:  "
        f"{type(model.layerwise_attention).__name__}"
    )

if hasattr(model, "estimator"):
    print(
        f"Estimator:            "
        f"{type(model.estimator).__name__}"
    )


# ============================================================
# Parameters per component
# ============================================================

print("\n" + "=" * 70)
print("PARAMETERS PER COMPONENT")
print("=" * 70)

for component_name, component in model.named_children():

    total_parameters = sum(
        parameter.numel()
        for parameter in component.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in component.parameters()
        if parameter.requires_grad
    )

    print(
        f"{component_name:<25} "
        f"{trainable_parameters:>12,} trainable / "
        f"{total_parameters:>12,} total"
    )


# ============================================================
# Total parameters
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print("\n" + "=" * 70)
print("TOTAL PARAMETERS")
print("=" * 70)

print(
    f"Trainable:            {trainable_parameters:,}"
)

print(
    f"Total:                {total_parameters:,}"
)

print(
    f"Trainable percentage: "
    f"{100 * trainable_parameters / total_parameters:.4f}%"
)

# ============================================================
# Loss and evaluation metrics
# ============================================================

print("\n" + "=" * 70)
print("LOSS AND METRICS")
print("=" * 70)

print(
    f"Configured loss:      "
    f"{getattr(model.hparams, 'loss', 'N/A')}"
)

if hasattr(model, "sentloss"):
    print(
        f"Sentence loss:        "
        f"{type(model.sentloss).__name__}"
    )

if hasattr(model, "train_corr"):
    print(
        f"Training metrics:     "
        f"{type(model.train_corr).__name__}"
    )

if hasattr(model, "word_level"):
    print(
        f"Word-level training:  "
        f"{model.word_level}"
    )