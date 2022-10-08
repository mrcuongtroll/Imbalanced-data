import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
import math


# Constants
UNDER_SAMPLING = ('u', 'under', 'us', 'under_sampling', 'under-sampling', 'under sampling')
OVER_SAMPLING = ('o', 'over', 'os', 'over_sampling', 'over-sampling', 'over sampling')


# Logging
logger = logging.getLogger(name=__name__)
# logger.propagate = False
# stream_handler = logging.StreamHandler()
# file_handler = logging.FileHandler('./logs/log_old.log')
# formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
# stream_handler.setFormatter(formatter)
# stream_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)
# file_handler.setLevel(logging.DEBUG)
# logger.addHandler(stream_handler)
# logger.addHandler(file_handler)
# logger.setLevel(logging.NOTSET)


# Classes
class BaseDataset(Dataset):
    def __init__(self, data: pd.DataFrame, label: str, sampling_mode: str = None):
        """
        Take a pandas DataFrame and create a pytorch Dataset. The Dataset can be resampled to balance the classes.
        :param data: type: pd.DataFrame. (Required) A pandas DataFrame storing the dataset, containing both the data and
                     the label columns.
        :param label: type: str. (Required) The name of the label column.
        :param sampling_mode: type: str. (Optional) Can be either over-sampling or under-sampling. If not given, no
                              resampling is applied.
        """
        super(BaseDataset, self).__init__()
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
def make_datasets(data: pd.DataFrame, label: str, splits=(0.7, 0.2, 0.1), batch_size=64,
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
    :param sampling_mode: type: str (Optional). See BaseDataset.sampling_mode.
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
        train_set = BaseDataset(data=train_df, label=label, sampling_mode=sampling_mode)
        logger.debug("Creating test set...")
        test_set = BaseDataset(data=test_df, label=label, sampling_mode=sampling_mode)
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
        train_set = BaseDataset(data=train_df, label=label, sampling_mode=sampling_mode)
        logger.debug("Creating validation set...")
        dev_set = BaseDataset(data=dev_df, label=label, sampling_mode=sampling_mode)
        logger.debug("Creating test set...")
        test_set = BaseDataset(data=test_df, label=label, sampling_mode=sampling_mode)
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


def make_dataloader(data: pd.DataFrame, label: str, batch_size=64, sampling_mode=None, copy=True, num_workers=0):
    """
    This function takes a pd.DataFrame and return a PyTorch DataLoader
    :param data: type: pd.DataFrame (Required). A DataFrame containing both data and labels.
    :param label: type: str (Required). The name of the label column.
    :param batch_size: type: int (Required. Default: 64). The batch size to create data loaders
    :param sampling_mode: type: str (Optional). See BaseDataset.sampling_mode.
    :param copy: type: boolean (Required. Default: True). Whether to copy the input DataFrame or to modify it inplace.
    :param num_workers: type: int (Required. Default: 0). The number of workers for each dataloaders.
    :return: DataLoader object
    """
    if copy:
        data = data.copy()
    logger.debug("Creating dataset...")
    dataset = BaseDataset(data=data, label=label, sampling_mode=sampling_mode)
    data_loader = BaseDataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
    return data_loader
