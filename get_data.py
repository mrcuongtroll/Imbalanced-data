"""
Run this script to download dataset from the internet and create metadata for the downloaded set.
"""

from data_loader.data_utils import *
import argparse


# Constants
DATASET_GET_FUNC = {'creditcard': get_creditcard,
                    'mnist': get_mnist}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get dataset from the internet and create metadata.")
    parser.add_argument("-d", "--dataset", type=str, help="Name of the dataset to get. To view the list of available "
                                                          "datasets, re-run this script with argument -ls or --list")
    parser.add_argument("-ls", "--list", default=False, action="store_true", help="View the list of available datasets")
    args = parser.parse_args()
    if args.list:
        print(f'Available datasets to get:\n{tuple(DATASET_GET_FUNC.keys())}')
        exit()
    assert args.dataset is not None, "-d/--dataset argument is required. " \
                                     "To view the list, re-run this script with argument -ls or --list."
    DATASET_GET_FUNC[args.dataset]()
