
import json
import argparse
from pathlib import Path
 
import pandas as pd
import torch
 
from comet import download_model, load_from_checkpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
 
from config import HF_CACHE_DIR
 
from pytorch_lightning import (
    Trainer,
    seed_everything,
)



# ============================================================
# Configuration file
# ============================================================
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file",type=str, required=True, help="Path to the experiment configuration file")
    return parser.parse_args()

def get_experiment_config(config_file):

    if not Path(config_file).exists():
        raise FileNotFoundError(
            f"Configuration file not found:\n"
            f"{config_file}"
        )
    with open(config_file, "r", encoding="utf-8") as file: 
        experiment_config = json.load(file)
    
    return( experiment_config["experiment_name"], 
           experiment_config["batch_size"], 
           experiment_config["learning_rate"], 
           experiment_config["max_epochs"], 
           experiment_config["devices"], 
           experiment_config["precision"], 
           experiment_config["trainable_modules"], 
           experiment_config["data_dir"], 
           experiment_config["output_dir"], 
           experiment_config.get("val_check_interval", 0.1)) 

def get_files(data_dir, output_dir, experiment_name): 
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    TRAIN_FILE = (Path(data_dir)/ "train_comet.csv")
    
    VALID_FILE = (Path(data_dir)/ "validation_comet.csv")
    
    OUTPUT_FILE = (Path(output_dir)/ f"comet_{experiment_name}.ckpt")
    
    
    if not TRAIN_FILE.exists():
    
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    
 
    if not VALID_FILE.exists():
    
        raise FileNotFoundError(f"Validation file not found: {VALID_FILE}")
    return TRAIN_FILE, VALID_FILE, OUTPUT_FILE


def print_experiment_info(EXPERIMENT_NAME, BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, DEVICES, PRECISION, TRAINABLE_MODULES, DATA_DIR, TRAIN_FILE, VALID_FILE, OUTPUT_FILE, VAL_CHECK_INTERVAL):
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)

    print("Experiment:",EXPERIMENT_NAME)

    print("Batch size:",BATCH_SIZE)

    print("Learning rate:",LEARNING_RATE)

    print("Epochs:",MAX_EPOCHS)

    print("Trainable modules:",TRAINABLE_MODULES)

    print("Devices:",DEVICES)

    print("Precision:",PRECISION)

    print("Validation check interval:",VAL_CHECK_INTERVAL)

    print("Data directory:",DATA_DIR)

    print("Train file:",TRAIN_FILE)

    print("Validation file:",VALID_FILE)

    print("Output file:",OUTPUT_FILE)


 
class ValHistory(Callback):
 
    def __init__(self):
        self.history = []
 
    def on_validation_end(self, trainer, pl_module):
 
        # Lightning runs a validation "sanity check" before training starts.
        # Skip it so it doesn't pollute the history with a junk first row.
        if trainer.sanity_checking:
            return
 
        metrics = trainer.callback_metrics
 
        row = {"step": trainer.global_step}
 
        for key in ["val_pearson", "val_kendall", "val_spearman"]:
            if key in metrics:
                row[key] = float(metrics[key])
 
        self.history.append(row)
 
 
val_history = ValHistory()
 
 
def get_gpus():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")

    available_gpus = torch.cuda.device_count()

    print("\nAvailable GPUs:",available_gpus)

    for gpu_index in range(available_gpus):
        print(f"GPU {gpu_index}:",torch.cuda.get_device_name(gpu_index))

    return available_gpus
 
 

def train_model(train_file, valid_file,devices,precision,max_epoch, experiment_name, output_file, batch_size, learning_rate,val_check_interval, trainable_modules):

    print("\nLoading COMET-Kiwi...")  
    model_path = download_model("Unbabel/wmt22-cometkiwi-da", saving_directory=HF_CACHE_DIR)    
    model = load_from_checkpoint(model_path)    
    print("Model loaded.")

   
    model.hparams.train_data = [str(train_file)]
    model.hparams.validation_data = [str(valid_file)]
    # Initialize model metrics after setting the training and validation data
    model.init_metrics() 
    
    
    for param in model.parameters():
    
        param.requires_grad = False
    

    for module_name in trainable_modules:
    
        if not hasattr(model, module_name):
    
            raise ValueError(
                f"Model does not contain module: "
                f"{module_name}"
            )
    
    
        module = getattr(model, module_name)
  
        for param in module.parameters():
    
            param.requires_grad = True
    
    

    model.nr_frozen_epochs = 0.0
    model.hparams.nr_frozen_epochs = 0.0
    model._frozen = False
    model.hparams.batch_size = (batch_size)
    model.hparams.learning_rate = (learning_rate)
 
 

    trainable = 0
    total = 0
 
    
    print("\n" + "=" * 70)
    print("TRAINABLE PARAMETERS")
    print("=" * 70)
    
 
    for name, param in model.named_parameters():
    
        total += param.numel()
    
        if param.requires_grad:
    
            trainable += param.numel()
    
            print(name)
    
    
    print(f"\nTrainable: {trainable:,}")
    
    print(f"Total: {total:,}")
    
    print(f"Trainable %: {100 * trainable / total:.4f}%")
 

    strategy = ("ddp" if devices > 1 else "auto")
    
    

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epoch,
        enable_progress_bar=False,
        enable_checkpointing=False,
        log_every_n_steps=50,
        logger=False,                 
        val_check_interval=val_check_interval,
        callbacks=[val_history]
    )
    

    print(f"\nStarting experiment:{experiment_name}")
    
    
    trainer.fit(model)
    
    trainer.save_checkpoint(output_file)
    
    

    print("\n" + "=" * 70)
    print("VALIDATION HISTORY")
    print("=" * 70)
    
    history_df = pd.DataFrame(val_history.history)
    
    print(
        history_df.to_string(index=False)
    )
    

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    
    print(
        "Model saved to:",
        output_file.resolve()
    )





def main():
    args = get_args()
    config_file = Path(args.config_file)

    EXPERIMENT_NAME, BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, DEVICES, PRECISION, TRAINABLE_MODULES, DATA_DIR, OUTPUT_DIR, VAL_CHECK_INTERVAL = get_experiment_config(config_file)
    if not Path(DATA_DIR).exists():
        raise FileNotFoundError(f"Data directory not found:{DATA_DIR}")

    TRAIN_FILE, VALID_FILE, OUTPUT_FILE = get_files(DATA_DIR, OUTPUT_DIR, EXPERIMENT_NAME)
 
    print_experiment_info(
        EXPERIMENT_NAME,
        BATCH_SIZE,
        LEARNING_RATE,
        MAX_EPOCHS,
        DEVICES,
        PRECISION,
        TRAINABLE_MODULES,
        DATA_DIR,
        TRAIN_FILE,
        VALID_FILE,
        OUTPUT_FILE,
        VAL_CHECK_INTERVAL
    )
    available_gpus = get_gpus()

    if DEVICES > available_gpus:
        raise RuntimeError(
            f"Requested {DEVICES} GPUs "
            f"but only {available_gpus} are available."
        )
    train_model(
        train_file=TRAIN_FILE,
        valid_file=VALID_FILE,
        devices=DEVICES,
        precision=PRECISION,
        max_epoch=MAX_EPOCHS,
        experiment_name=EXPERIMENT_NAME,
        output_file=OUTPUT_FILE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        val_check_interval=VAL_CHECK_INTERVAL,
        trainable_modules=TRAINABLE_MODULES
    )   


if __name__ == "__main__":
    main()

