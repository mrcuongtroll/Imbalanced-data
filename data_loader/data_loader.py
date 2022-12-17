import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
import math
import os
import json
from definitions import *


# datasets path
if not os.path.exists(os.path.abspath(DATASETS_DIR)):
    os.makedirs(os.path.abspath(DATASETS_DIR))
if not os.path.exists(os.path.abspath(DATASETS_METADATA_DIR)):
    os.makedirs(os.path.abspath(DATASETS_METADATA_DIR))


# Constants
UNDER_SAMPLING = ('u', 'under', 'us', 'under_sampling', 'under-sampling', 'under sampling')
OVER_SAMPLING = ('o', 'over', 'os', 'over_sampling', 'over-sampling', 'over sampling')


# Logging
logger = logging.getLogger(name=__name__)


# Classes
class DFDataset(Dataset):
    def __init__(self, data: pd.DataFrame, label: str, sampling_mode: str = None):
        """
        Take a pandas DataFrame and create a pytorch Dataset. The Dataset can be resampled to balance the classes.
        :param data: type: pd.DataFrame. (Required) A pandas DataFrame storing the dataset, containing both the data and
                     the label columns.
        :param label: type: str. (Required) The name of the label column.
        :param sampling_mode: type: str. (Optional) Can be either over-sampling or under-sampling. If not given, no
                              resampling is applied.
        """
        super(DFDataset, self).__init__()
        assert label in data.columns, f"'{label}' is not a column name of the given DataFrame."
        assert data[label].dtype in (np.int64, np.int32, np.int), "Class labels must be integers."
        if sampling_mode is None:
            label_values = data[label].unique()
            logger.debug(f"Samples per class:")
            for value in label_values:
                logger.debug(f"Class: {value}.\tSamples: {len(data[data[label] == value])}")
        elif sampling_mode.lower() in UNDER_SAMPLING:
            logger.debug("Applying resampling mode: Under-sampling.")
            label_values = data[label].unique()
            # Find the minimum number of samples from a class
            min_count = np.inf
            for value in label_values:
                if len(data[data[label] == value]) < min_count:
                    min_count = len(data[data[label] == value])
            logger.debug("Samples per class: ")
            # Resample so that every class has the same number of sample as min_count
            classes = []
            for value in label_values:
                new_class_subset = data[data[label] == value].sample(frac=min_count/len(data[data[label] == value]))
                logger.debug(f"Class: {value}.\tSamples: {len(new_class_subset[new_class_subset[label] == value])}")
                classes.append(new_class_subset)
            data = pd.concat(classes)
        elif sampling_mode.lower() in OVER_SAMPLING:
            logger.debug("Applying resampling mode: Over-sampling.")
            label_values = data[label].unique()
            # Find the maximum number of samples from a class
            max_count = 0
            for value in label_values:
                if len(data[data[label] == value]) > max_count:
                    max_count = len(data[data[label] == value])
            logger.debug("Samples per class: ")
            # Resample so that every class has the same number of sample as max_count
            classes = []
            for value in label_values:
                new_class_subset = pd.concat([data[data[label] == value]] * round(max_count / len(data[data[label] == value])))
                logger.debug(f"Class: {value}.\tSamples: {len(new_class_subset[new_class_subset[label] == value])}")
                classes.append(new_class_subset)
            data = pd.concat(classes)
        else:
            raise ValueError("Invalid resampling method. Please choose either over-sampling, under-sampling, or None.")
        labels = data.pop(label)
        self.data = torch.FloatTensor(data.values)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def plot_percentage(self, title='Percentage of data classes', class_names=None, ax=None):
        labels = self.labels.detach().cpu().numpy()
        classes = np.sort(np.unique(labels))
        class_count = []
        class_percentage = []
        for c in classes:
            count = np.sum(labels == c)
            class_count.append(count)
            class_percentage.append(100 * count / len(labels))
        if ax is None:
            fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(7, 7))
        if class_names is None:
            class_names = classes
        ax.set_title(title, size=20)
        ax.bar(class_names, class_percentage)
        count_iter = iter(class_count)
        for bar in ax.patches:
            ax.annotate(str(next(count_iter)),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='center', size=15, xytext=(0, 8), textcoords='offset points')
        return class_count


class TVDataset(Dataset):

    def __init__(self, dataset_name: str, train: bool = True, flatten_images: bool = False,
                 sampling_mode: str = None, make_imbalanced=True,
                 classes=(0, 1), imbalanced_ratio: float = 0.05, minority_classes=(1,)):
        """
        Take a torchvision VisionDataset class name and create a pytorch Dataset. The dataset will be made imbalanced
        and can then be further resampled to balance the (artificially imbalanced) classes.
        :param data: type: type(torchvision.datasets.VisionDataset). (Required) A PyTorch built-in vision dataset that
                     will be downloaded and customized.
        :param train: type: bool. (Required, default=True). Whether to download the training set or the test set.
        :param flatten_images: type: bool. (Required, default=False) Whether to flatten images in the vision dataset or
                               not.
        :param sampling_mode: type: str. (Optional) Can be either over-sampling or under-sampling. If not given, no
                              resampling is applied.
        :param classes: type: tuple(int). (Required, default=(0, 1)) The classes to read from the dataset
        :param imbalanced_ratio: type float. (Required, default=0.05) The imbalance ratio, i.e. the number of minority
                                 class samples divided by the number of majority class samples.
        :param minority_classes: type: tuple(int). (Required, default=(1,)) The minority classes.
        """
        super(TVDataset, self).__init__()
        assert os.path.exists(os.path.join(DATASETS_METADATA_DIR, f'{dataset_name}.json')), "Metadata for MNIST not found. " \
                                                                                  "Please create one or run " \
                                                                                  "python get_data.py -d=mnist"
        with open(os.path.join(DATASETS_METADATA_DIR, f'{dataset_name}.json'), 'r') as f:
            metadata = json.load(f)
        data = DATASETS_MAP[dataset_name]
        self.dataset = data(
            root=DATASETS_DIR, train=train, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        self.imbalanced = make_imbalanced
        self.classes = classes                      # Dunno what to do with this yet
        self.minority_classes = minority_classes    # Dunno what to do with this yet
        self.imbalanced_ratio = imbalanced_ratio    # Dunno what to do with this yet
        dset = 'train' if train else 'test'
        num_minor_samples = round(imbalanced_ratio * max([metadata["num_samples_per_class"][dset][str(c)]
                                                          for c in classes if c not in minority_classes]))
        # Get indices from the original dataset to create artificially imbalanced dataset
        self.data_indices = []   # This list stores indices of the samples to actually be used.
        self.sample_count = {label: 0 for label in classes}     # used to limit the number of minority samples
        for idx in range(len(self.dataset)):
            label = self.dataset[idx][1]
            if label in classes:
                if label in minority_classes and self.sample_count[label] >= num_minor_samples and self.imbalanced:
                    continue
                self.data_indices.append(idx)
                self.sample_count[label] += 1
        logger.info(f"Dataset sample count:")
        for c in self.sample_count.keys():
            logger.info(f"Class: {c}. Sample count: {self.sample_count[c]}")
        self._len = len(self.data_indices)
        self.flatten_images = flatten_images
        self.sampling_mode = sampling_mode

    def __len__(self):
        # return len(self.data_indices)   # O(n) I guess...
        return self._len   # O(1)

    def __getitem__(self, idx):
        if idx >= self._len:
            raise IndexError(f"Index out of range for length {self._len}")
        true_idx = self.data_indices[idx]
        data, target = self.dataset[true_idx]
        if self.flatten_images:
            logger.info(torch.flatten(data).shape)
            return torch.flatten(data), torch.tensor(target, dtype=torch.long)
        else:
            return data, torch.tensor(target, dtype=torch.long)


class BaseDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def sample(self, sample_size: int):
        """
        Sample a random batch from the dataset. (TODO)
        :param sample_size: type: int (Required). The size of the random batch to sample.
        :return: data: Tensor, target: Tensor
        """
        return


# Functions
def make_tv_dataloaders(dataset_name: str, dev_split: float = None, batch_size: int = 64,
                        flatten_images: bool = False, sampling_mode: str = None, num_workers: int = 0):
    """
    This function takes a dataset class name from torchvision.datasets and return data loaders of train, dev and test
    splits of said dataset.
    :param dataset_name: type: type(torchvision.datasets.VisionDataset). (Required) A PyTorch built-in vision dataset that will
                 be downloaded and customized.
    :param dev_split: type: float. (Optional) The ratio to split the training set into a new training set and a
                      validation set. If not given, the training set will not be split.
    :param batch_size: type: int. (Required, default=64) The batch size of the data loaders.
    :param flatten_images: type: bool. (Required, default=False) Whether or not to flatten the images.
    :param sampling_mode: type: str. (Optional) Can be either over-sampling or under-sampling. If not given, no
                          resampling is applied.
    :param num_workers: type: int. (Required, default=0) The parameter num_workers to be passed to torch DataLoader.
    :return: (train: DataLoader, dev: DataLoader, test: DataLoader) or
             (train: DataLoader, test: DataLoader)
    """
    train_set = TVDataset(dataset_name, train=True, flatten_images=flatten_images, sampling_mode=sampling_mode)
    dev_set = None
    if dev_split is not None:
        train_set, dev_set = train_test_split(train_set, test_size=dev_split)
    test_set = TVDataset(dataset_name, train=False, flatten_images=flatten_images, sampling_mode=sampling_mode,
                         make_imbalanced=False)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers)
    if dev_set is not None:
        dev_loader = DataLoader(dataset=dev_set,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
        return train_loader, dev_loader, test_loader
    else:
        return train_loader, test_loader
    

def make_df_datasets(data: pd.DataFrame, label: str, splits=(0.7, 0.2, 0.1), batch_size=64,
                     sampling_mode=None, return_dframe=False, copy=True, num_workers=0):
    """
    This function takes a pandas DataFrame as input and return train/dev/test splits as PyTorch DataLoaders.
    This should be useful if you only have pre-split dataset.
    :param data: type: pd.DataFrame (Required). A DataFrame containing both data and labels.
    :param label: type: str (Required). The name of the label column.
    :param splits:  type: tuple(float) (Required. Default: (0.7, 0.2, 0.1)). Splitting ratio. If len(splits)==2 then
                    split the DataFrame into training and test sets according to the ratios. If len(splits)==3 then split
                    the DataFrame into training, development and test sets according to the ratios.
    :param batch_size: type: int (Required. Default: 64). The batch size to create data loaders
    :param sampling_mode: type: str (Optional). See DFDataset.sampling_mode.
    :param return_dframe: type: bool (Required. Default: False). If set to True, this function will return the datasets
                          as pd.DataFrame's. Otherwise, it will return PyTorch DataLoader's.
    :param copy: type: boolean (Required. Default: True). Whether to copy the input DataFrame or to modify it inplace.
    :param num_workers: type: int (Required. Default: 0). The number of workers for each dataloaders.
    :return: (train: DataLoader, dev: DataLoader, test: DataLoader) or
             (train: DataLoader, test: DataLoader)
    """
    assert len(splits) == 3 or len(splits) == 2, "splits must be a tuple of either 2 or 3 floats."
    assert math.isclose(sum(splits), 1.0), "The splitting ratios must add up to 1."
    assert label in data.columns, f"'{label}' is not a column of the DataFrame."
    if copy:
        data = data.copy()
    orig_labels = data.pop(label)
    logger.info(f"Making datasets... Label column: '{label}'. Splits ratio: {splits}. Batch size: {batch_size}. "
                f"Sampling mode: {sampling_mode}. Copy: {copy}.")
    if len(splits) == 2:
        logger.info(f"Splitting train/test sets. Ratio: {splits}.")
        data_train, data_test, label_train, label_test = train_test_split(data, orig_labels, test_size=splits[1],
                                                                          stratify=orig_labels)
        train_df = pd.concat([data_train, label_train], axis=1)
        test_df = pd.concat([data_test, label_test], axis=1)
        if return_dframe:
            return train_df, test_df
        logger.debug("Creating training set...")
        train_set = DFDataset(data=train_df, label=label, sampling_mode=sampling_mode)
        logger.debug("Creating test set...")
        test_set = DFDataset(data=test_df, label=label, sampling_mode=sampling_mode)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
        return train_loader, test_loader
    elif len(splits) == 3:
        logger.info(f"Splitting train/dev/test sets. Ratio: {splits}.")
        data_train, data_test, label_train, label_test = train_test_split(data, orig_labels, test_size=splits[2],
                                                                          stratify=orig_labels)
        data_train, data_dev, label_train, label_dev = train_test_split(data_train, label_train,
                                                                        test_size=splits[1]/(1-splits[2]),
                                                                        stratify=label_train)
        train_df = pd.concat([data_train, label_train], axis=1)
        dev_df = pd.concat([data_dev, label_dev], axis=1)
        test_df = pd.concat([data_test, label_test], axis=1)
        if return_dframe:
            return train_df, dev_df, test_df
        logger.debug("Creating training set...")
        train_set = DFDataset(data=train_df, label=label, sampling_mode=sampling_mode)
        logger.debug("Creating validation set...")
        dev_set = DFDataset(data=dev_df, label=label, sampling_mode=sampling_mode)
        logger.debug("Creating test set...")
        test_set = DFDataset(data=test_df, label=label, sampling_mode=sampling_mode)
        train_loader = BaseDataLoader(dataset=train_set,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)
        dev_loader = BaseDataLoader(dataset=dev_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)
        test_loader = BaseDataLoader(dataset=test_set,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)
        return train_loader, dev_loader, test_loader


def make_df_dataloader(data: pd.DataFrame, label: str, batch_size=64, sampling_mode=None, copy=True, num_workers=0):
    """
    This function takes a pd.DataFrame and return a PyTorch DataLoader
    :param data: type: pd.DataFrame (Required). A DataFrame containing both data and labels.
    :param label: type: str (Required). The name of the label column.
    :param batch_size: type: int (Required. Default: 64). The batch size to create data loaders
    :param sampling_mode: type: str (Optional). See DFDataset.sampling_mode.
    :param copy: type: boolean (Required. Default: True). Whether to copy the input DataFrame or to modify it inplace.
    :param num_workers: type: int (Required. Default: 0). The number of workers for each dataloaders.
    :return: DataLoader object
    """
    if copy:
        data = data.copy()
    logger.debug("Creating dataset...")
    dataset = DFDataset(data=data, label=label, sampling_mode=sampling_mode)
    data_loader = BaseDataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
    return data_loader
