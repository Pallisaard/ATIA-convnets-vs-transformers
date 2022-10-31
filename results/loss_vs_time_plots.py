import os
from typing import Any, Callable, List, Tuple, TypeVar

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


T = TypeVar("T")


def plot_loss_over_time(log_df: pd.DataFrame,
                        model: str,
                        dataset: str,
                        save_path: str) -> None:
    """
    Function that plots accuracy over time according to cell above, saves to
    results folder and shows it
    args:
        log_df: pd.DataFrame containing the logs
        model: str containing model name
        dataset: str containing dataset name
    """
    # Get row with minimum val_loss
    val_stats = log_df.query("val_loss.notna()")
    train_loss = log_df.query("train_loss_epoch.notna()")
    plot_df = val_stats.copy()
    plot_df["train_loss"] = train_loss["train_loss_epoch"].values
    plot_df = plot_df[["epoch", "val_loss", "val_acc", "train_loss"]]
    plot_df.epoch = plot_df.epoch + 1

    # Set figure size to 7x4
    plt.figure(figsize=(7, 4))
    # Plot plt line of epoch vs val_loss
    plt.plot(plot_df.epoch, plot_df.val_loss, label="val_loss")
    # Plot plt line of epoch vs train_loss
    plt.plot(plot_df.epoch, plot_df.train_loss, label="train_loss")
    # Get epoch where val_loss is minimum
    val_loss_min_idx = plot_df.val_loss.idxmin()
    epoch_with_min_loss = plot_df.loc[val_loss_min_idx].epoch + 1
    # Add dotted black vertical line at epoch with minimum val_loss with 50%
    # opacity labeled "min val loss"
    plt.axvline(epoch_with_min_loss, color="black", linestyle="--", alpha=0.5, label="min val loss")

    # Set x and y labels
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # Add legend
    plt.legend()
    # Add title with large font size
    plt.title(f"Loss over time for {model} on {dataset}", fontsize=16)
    # Save figure as png with dpi 300
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Show figure
    plt.show()


# Recursively calls os.path.dirname until it reaches the given directory
def dirname_until(dirname: str, until: str) -> str:
    """Recursively calls os.path.dirname until it reaches the given directory

    Args:
        dirname (str): The directory to start from
        until (str): The folder to stop at

    Returns:
        str: The directory that was reached
    """

    while dirname != "":
        if os.path.basename(dirname) == until:
            return dirname
        dirname = os.path.dirname(dirname)
    return ""


def recursive_call(apply: Callable[[T], T], on: T, times: int) -> T:
    """Calls apply on a variable 'on' 'times' times

    Args:
        apply (Callabel[[T], T]): A function from generic T to generic T
        on (T): A variable of type T
        times (int): The number of times to apply the function

    Returns:
        T: The result of applying the function 'times' times
    """
    val = on
    for _ in range(times):
        val = apply(val)
    return val


# Function unzip(lst) that takes a list of tuples and unzips it into two lists
def unzip(lst: List[Tuple]) -> Tuple[List[Any], List[Any]]:
    return [x[0] for x in lst], [x[1] for x in lst]


def head(lst: List, n=1) -> List:
    return lst[:n]


def get_log_file_paths() -> List[str]:
    # Use os to get all files ending with csv in the ../experiments folder
    log_files = [
        os.path.join(root, name)
        for root, dirs, files in os.walk("../experiments")
        for name in files if name.endswith(".csv")
    ]
    return log_files


def get_model_dataset_version_tuple(log_files: List[str]) -> List[Tuple[str, str, str]]:
    # Extract model name and dataset name from file path
    # eg. "../experiments/convnext-isic_2019/9/logs/convnext/version_0/metrics.csv"
    # -> "convnext", "isic_2019"
    model_dataset_pairs = [
        os.path.basename(recursive_call(os.path.dirname, file, times=5)).split("-")
        for file in log_files
    ]
    model_dataset_pairs = [tuple(pair) for pair in model_dataset_pairs]
    # Extract experiment version from file path
    # eg. "../experiments/convnext-isic_2019/9/logs/convnext/version_0/metrics.csv" -> 9
    experiment_versions = [
        os.path.basename(recursive_call(os.path.dirname, file, times=4))
        for file
        in log_files
    ]
    # For each file, join model name, dataset name and experiment version into a single tuple
    model_dataset_version = list(zip(*unzip(model_dataset_pairs), experiment_versions))
    return model_dataset_version


def main() -> None:
    log_files = get_log_file_paths()
    # Create folder for plots if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")
    # Get model, dataset and version tuple
    model_dataset_version = get_model_dataset_version_tuple(log_files)
    # Loop through each model, dataset and version.
    # For each, load the logs and plot the loss over time
    for model, dataset, version in model_dataset_version:
        log_df = pd.read_csv(os.path.join(
            "../experiments",
            f"{model}-{dataset}",
            version,
            "logs",
            model,
            "version_0",
            "metrics.csv"
        ))
        plot_loss_over_time(log_df,
                            model,
                            dataset,
                            os.path.join("plots", f"{model}-{dataset}-{version}.png"))


if __name__ == "__main__":
    main()
