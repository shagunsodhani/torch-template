from typing import Dict

import torch
import torchvision
from torchvision import transforms as transforms

from codes.dataset.types import DataLoaderType
from codes.utils.config import ConfigType


def get_dataloaders(config: ConfigType) -> Dict[str, DataLoaderType]:
    dataset_config = config.dataset
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=dataset_config.dir, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, **dataset_config["dataloader"]["train"]
    )

    testset = torchvision.datasets.CIFAR10(
        root=dataset_config.dir, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, **dataset_config["dataloader"]["test"]
    )

    return {
        "train": trainloader,
        "test": testloader,
    }
