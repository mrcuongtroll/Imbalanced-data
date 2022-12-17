"""
This file contains constants frequently used by other modules.
"""


import os
from model import GrowingNN as Grow
from torchvision import datasets as datasets


# CONSTANTS:
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
DATASETS_METADATA_DIR = os.path.join(DATASETS_DIR, 'metadata')
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
DATASETS_MAP = {'mnist': datasets.MNIST,
                'cifar10': datasets.CIFAR10,
                'cifar100': datasets.CIFAR100}
ARCHITECTURES_MAP = {
    'mlp': Grow.GrowingMLP,
    'cnn': Grow.GrowingCNN
}
FLATTEN_FOR_ARCH = {
    'mlp': True
    'cnn': False
}


if __name__ == '__main__':
    # TEST PURPOSE ONLY
    print(ROOT_DIR)
