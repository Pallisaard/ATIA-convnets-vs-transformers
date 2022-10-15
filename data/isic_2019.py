from typing import Optional, Tuple, List
from os import path

import PIL
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
import pandas as pd


def _create_isic_ground_truth_table(filepath: str,
                                    save_result: bool = True,
                                    save_path: Optional[str] = None
                                    ) -> pd.DataFrame:
    isic_labels = pd.read_csv(filepath)

    # Format the data into a table having the image name, the label name and the label
    image_names = isic_labels.iloc[:, 0]
    image_labels = isic_labels.iloc[:, 1:].idxmax(axis=1)
    isic_label_info = pd.concat(
        [image_names, image_labels], axis=1
    ).rename(columns={0: "label_name"})
    isic_label_info["label"] = isic_label_info["label_name"].map(
        {"MEL": 0, "NV": 1, "BCC": 2, "AK": 3, "BKL": 4, "DF": 5, "VASC": 6, "SCC": 7, "UNK": 8}
    ).astype(np.int32)

    # keep only the first 3000 examples of label 1 (NV) from the dataset isic_label_info
    is_label_1 = isic_label_info[isic_label_info["label"] == 1]
    is_not_label_1 = isic_label_info[isic_label_info["label"] != 1]
    random_label_1_indices = np.random.randint(0, len(is_label_1), 3000)
    isic_label_info_tmp = pd.concat([is_not_label_1, is_label_1.iloc[random_label_1_indices, :]])

    # keep only the first 3000 examples of label 0 (MEL) from the dataset isic_label_info
    is_label_0 = isic_label_info_tmp[isic_label_info_tmp["label"] == 0]
    is_not_label_0 = isic_label_info_tmp[isic_label_info_tmp["label"] != 0]
    random_label_0_indices = np.random.randint(0, len(is_label_0), 3000)
    isic_label_info_balanced = pd.concat([
        is_not_label_0,
        is_label_0.iloc[random_label_0_indices, :]]
    )

    # Create sample probabilities
    balanced_label_counts = (
        isic_label_info_balanced.drop(columns=["image"])
                                .groupby(["label"])
                                .count()["label_name"]
                                .values)
    highest_label_count = np.repeat(np.max(balanced_label_counts), len(balanced_label_counts))
    sample_ratios = highest_label_count / balanced_label_counts
    sample_probabilities = sample_ratios / np.sum(sample_ratios)

    isic_gt_table = isic_label_info_balanced.copy()
    isic_gt_table["sample_prob"] = sample_probabilities[isic_gt_table["label"].values]

    if save_result and save_path is not None:
        isic_gt_table.to_csv(save_path, index=False)

    return isic_gt_table


class ISIC2019Dataset(Dataset):
    def __init__(self, root: str, transform: nn.Module = None):
        self.root: str = root
        self.ground_truth = root + "/isic_2019_ground_truth.csv"
        self.transform: T.Compose = transform

        self.gtdata = _create_isic_ground_truth_table(
            filepath=self.root + "/ISIC_2019_Training_GroundTruth.csv",
            save_path=self.ground_truth
        )

        self.gt_data: pd.DataFrame = self._load_gt()
        self.sample_probs: torch.Tensor = torch.Tensor(self.gt_data["sample_prob"].values)

    def _load_gt(self):
        return pd.read_csv(self.ground_truth)

    def __getitem__(self, index):
        image_path = path.join(self.root, "images", self.gt_data.iloc[index]["image"] + ".jpg")
        if image_path[0] == "~":
            image_path = path.expanduser(image_path)

        image = PIL.Image.open(image_path)
        image = image.convert("RGB")
        label = self.gt_data.iloc[index]["label"]

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        return image, label

    def __len__(self) -> int:
        return len(self.gt_data)


def get_isic_2019_feature_extractor(image_size: Tuple[int, int] = (224, 224)) -> T.Compose:
    return T.Compose([
        T.PILToTensor(),
        T.Resize(image_size, T.InterpolationMode.BILINEAR, antialias=False),
        T.RandomCrop(image_size),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_isic_2019_data(root: str,
                       transform: T.Compose = None,
                       train_split: float = 0.8,
                       split_seed: int = 42) -> List[Dataset]:
    ds = ISIC2019Dataset(root, transform)
    lengths = [int(train_split * len(ds)), len(ds) - int(train_split * len(ds))]
    return torch.utils.data.random_split(
        ds,
        lengths=lengths,
        generator=torch.Generator().manual_seed(split_seed)
    )
