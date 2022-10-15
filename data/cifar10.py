from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T

DEFAULT_DATA_DIR = "datasets/"


def get_cifar10_train(root: str = DEFAULT_DATA_DIR,
                      transforms: T.Compose = None) -> Dataset:
    return datasets.CIFAR10(root=root,
                            train=True,
                            download=True,
                            transform=transforms)


def get_cifar10_test(root: str = DEFAULT_DATA_DIR,
                     transforms: T.Compose = None) -> Dataset:
    return datasets.CIFAR10(root=root,
                            train=False,
                            download=True,
                            transform=transforms)


def get_cifar10_data(root: str = DEFAULT_DATA_DIR,
                     train_transforms: T.Compose = None,
                     test_transforms: T.Compose = None) -> List[Dataset]:
    """

    :rtype: object
    """
    cifar10_train = get_cifar10_train(root, train_transforms)
    cifar10_test = get_cifar10_test(root, test_transforms)
    return [cifar10_train, cifar10_test]


def get_cifar10_feature_extractor(image_size: Tuple[int, int] = (224, 224)) -> T.Compose:
    return T.Compose([
        T.PILToTensor(),
        T.Resize(image_size, T.InterpolationMode.BILINEAR, antialias=False),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])