import torch
from torchvision import datasets
from torchvision import transforms as T

DEFAULT_DATA_DIR = "datasets/"


def get_cifar10_train(root=DEFAULT_DATA_DIR,
                      transforms=None):
    return datasets.CIFAR10(root=root,
                            train=True,
                            download=True,
                            transform=transforms)


def get_cifar10_test(root=DEFAULT_DATA_DIR,
                     transforms=None):
    return datasets.CIFAR10(root=root,
                            train=False,
                            download=True,
                            transform=transforms)


def get_cifar10_data(root=DEFAULT_DATA_DIR,
                     train_transforms=None,
                     test_transforms=None):
    cifar10_train = get_cifar10_train(root, train_transforms)
    cifar10_test = get_cifar10_test(root, test_transforms)
    return cifar10_train, cifar10_test

def get_cifar10_feature_extractor(image_size=(224, 224)):
    return T.Compose([
        T.PILToTensor(),
        T.Resize(image_size, T.InterpolationMode.BILINEAR, antialias=False),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])