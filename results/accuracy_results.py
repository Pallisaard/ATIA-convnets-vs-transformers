import os
from os import path
import pandas as pd
import argparse

print("current path:", os.getcwd())

MODEL_NAMES = ["convnext:cifar10", "convnext:isic_2019", "swin:cifar10", "swin:isic_2019"]
MODEL_VERSIONS = [str(i) for i in range(1, 11)]
EXPERIMENT_DIR = "../experiments"

def get_log_dir(experiment_dir: str,
                experiment_name: str,
                experiment_n: int) -> str:
    return path.join(experiment_dir, experiment_name, experiment_n, "logs")

def get_checkpoint_dir(experiment_dir: str,
                       experiment_name: str,
                       experiment_n: int) -> str:
    return path.join(experiment_dir, experiment_name, experiment_n, "checkpoints")

def get_log_file(log_dir: str) -> pd.DataFrame:
    return pd.read_csv(path.join(log_dir, "convnext", "version_0", "metrics.csv"))

def get_checkpoint_file_names(checkpoint_dir: str) -> list:
    return os.listdir(checkpoint_dir)

def get_epoch_from_checkpoint_file(checkpoint_file: str) -> int:
    return int(checkpoint_file.split("=")[1].split("-")[0])

def get_acc_from_log_file(log_file: pd.DataFrame,
                          epoch: int) -> float:
    return log_file.query(f"epoch == {epoch} and val_loss.notna()")["val_acc"].values[0]

def generate_accuracy_results_df(model_names: list,
                                 model_versions: list,
                                 experiment_dir: str) -> pd.DataFrame:
    results = []
    for model_name in model_names:
        for model_version in model_versions:
            log_dir = get_log_dir(experiment_dir, model_name, model_version)
            checkpoint_dir = get_checkpoint_dir(experiment_dir, model_name, model_version)
            try:
                log_file = get_log_file(log_dir)
                checkpoint_file_names = get_checkpoint_file_names(checkpoint_dir)
                for checkpoint_file in checkpoint_file_names:
                    epoch = get_epoch_from_checkpoint_file(checkpoint_file)
                    acc = get_acc_from_log_file(log_file, epoch)
                    
                    results.append({"model_name": model_name,
                                        "model_version": model_version,
                                        "epoch": epoch,
                                        "val_acc": acc})
            except FileNotFoundError:
                print(f"Warning - FileNotFoundError: {model_name} {model_version} {epoch}")
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    args = parser.parse_args()
    
    
    results = generate_accuracy_results_df(
        MODEL_NAMES,
        MODEL_VERSIONS,
        args.experiment_dir
    )
    results.to_csv("accuracy_results.csv", index=False)

if __name__ == "__main__":
    main()
