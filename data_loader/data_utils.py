import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd
import logging
import math
import os
import json
import wget
from definitions import *


# datasets path
if not os.path.exists(os.path.abspath(DATASETS_DIR)):
    os.makedirs(os.path.abspath(DATASETS_DIR))
if not os.path.exists(os.path.abspath(DATASETS_METADATA_DIR)):
    os.makedirs(os.path.abspath(DATASETS_METADATA_DIR))


# Logging
# logger = logging.getLogger(name=__name__)


# Functions
"""
get_<dataset_name>() functions below are used to download the corresponding dataset and save their metadata.
"""
def get_creditcard():
    file_name = os.path.join(DATASETS_DIR, 'creditcard.csv')
    if not os.path.exists(os.path.join(DATASETS_DIR, 'creditcard.csv')):
        url = 'https://datahub.io/machine-learning/creditcard/r/creditcard.csv'
        file_name = wget.download(url, out=DATASETS_DIR)
    if not os.path.exists(os.path.join(DATASETS_METADATA_DIR, 'creditcard.json')):
        data = pd.read_csv(file_name)
        metad = {"type": "csv",
                 "label_col": "Class",
                 "pop_cols": ["Time", "Amount"],
                 "num_features": 28,
                 "num_classes": 2,
                 "classes": (0, 1),
                 "num_samples": len(data),
                 "num_samples_per_class": {},
                 "model_args": {
                     "input_size": 28,
                     "output_size": 2
                    }
                 }
        label_values = data["Class"].unique()
        for label in label_values:
            metad["num_samples_per_class"][label] = len(data[data["Class"] == label])
        with open(os.path.join(DATASETS_METADATA_DIR, 'creditcard.json'), 'w') as f:
            json.dump(metad, f, indent=4)
    return file_name


def get_mnist():
    train_set = torchvision.datasets.MNIST(
            root=DATASETS_DIR, train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
    test_set = torchvision.datasets.MNIST(
            root=DATASETS_DIR, train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
    if not os.path.exists(os.path.join(DATASETS_METADATA_DIR, 'mnist.json')):
        metad = {"type": "torchvisionDataset",
                 "num_features": 28*28,
                 "num_classes": 10,
                 "classes": list(range(0, 10)),
                 "num_samples": {
                     "train": len(train_set),
                     "test": len(test_set)
                 },
                 "num_samples_per_class": {
                     "train": {i: 0 for i in range(0, 10)},
                     "test": {i: 0 for i in range(0, 10)},
                 },
                 "model_args": {
                     "input_size": 28*28,
                     "output_size": 10,
                    }
                 }
        for sample, label in train_set:
            metad["num_samples_per_class"]["train"][label] += 1
        for sample, label in test_set:
            metad["num_samples_per_class"]["test"][label] += 1
        with open(os.path.join(DATASETS_METADATA_DIR, 'mnist.json'), 'w') as f:
            json.dump(metad, f, indent=4)
    return


if __name__ == '__main__':
    # Test purpose only
    # print("\n", get_creditcard())
    get_mnist()
