import sys
import torch
import logging
import argparse
import pandas as pd
import numpy as np
import os
from data_loader.data_loader import make_datasets
from model.GrowingNN import GrowingMLP
from trainer.trainer import Trainer
from utils.utils import test_report


# Logging
logger = logging.getLogger()
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.setLevel(logging.NOTSET)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.exception("Uncaught exception encountered", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


def main(args):
    logger.info(f"Loading dataset from {args.csv}...")
    data = pd.read_csv(args.csv)
    try:
        data.pop("Time")
    except:
        pass
    logger.debug(f"Example samples from the dataset:\n{data.head()}")
    train_loader, dev_loader, test_loader = make_datasets(data=data, label=args.label_col,
                                                          batch_size=args.batch_size,
                                                          sampling_mode=None)
    train_loader_u, dev_loader_u, test_loader_u = make_datasets(data=data, label=args.label_col,
                                                                batch_size=args.batch_size,
                                                                sampling_mode='under-sampling')
    logger.info("... Done!")
    input_size = len(data.columns) - 1  # Minus the label column
    output_size = len(data[args.label_col].unique())
    logger.info(f"Creating model with input_size={input_size}, output_size={output_size}")
    model = GrowingMLP(input_size, output_size, device=args.device)
    logger.info("Initializing parameters states")
    model.init_params_state()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam
    epochs = args.epochs
    lr = args.learning_rate
    logger.info(f"Training GrowingMLP: Criterion={criterion} | Optimizer={optimizer} | Epochs={epochs} | lr={lr}")
    trainer = Trainer(model=model, optimizer=optimizer, learning_rate=args.learning_rate, device=args.device)
    trainer.train(train_loader=train_loader_u, dev_loader=dev_loader_u, gen_rate=args.gen_rate,
                  criterion=criterion, epochs=30*epochs, num_prints=args.num_prints, high=args.high, low=args.low)
    trainer.train(train_loader=train_loader, dev_loader=dev_loader, gen_rate=args.gen_rate,
                  criterion=criterion, epochs=epochs, num_prints=args.num_prints, high=args.high, low=args.low)
    test_report(model, test_loader)
    # logger.debug(f"linear4's weight: {model.linear4.weight}")
    # logger.debug(f"linear4's bias: {model.linear4.bias}")
    # logger.debug(f"linear4's weight mask: {model.params_state['linear4.weight']}")
    # logger.debug(f"linear4's bias mask: {model.params_state['linear4.bias']}")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GrowingNN test")
    parser.add_argument("--csv", type=str, required=True, help="Directory to the csv dataset. (Required).")
    parser.add_argument("--label_col", type=str, default="Class", help="The name of the label column. Default: 'Class'")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="The learning rate. Default: 0.0001")
    parser.add_argument("-e", "--epochs", type=int, default=3, help='The number of epochs to train. Default: 3'
                                                                    'The model will be trained for 30*epochs epochs '
                                                                    'on the under-sampling set before being trained for'
                                                                    ' epochs on the full set.')
    parser.add_argument("-b", "--batch_size", type=int, default=64, help='The batch size of split sets. Default: 64')
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Either 'cpu' or 'cuda'. Default: 'cuda'")
    parser.add_argument("--high", type=float, default=0.5, help="The upper threshold, used for neural freezing. "
                                                                "Default: 0.5")
    parser.add_argument("--low", type=float, default=0.25, help="The lower threshold, used for neural freezing. "
                                                                "Default: 0.25")
    parser.add_argument("--gen_rate", type=float, default=0.2, help="The neural generation rate. Default: 0.2. "
                                                                    "e.g. gen_rate=0.2 "
                                                                    "means that there will be 20%% of the original "
                                                                    "number of neurons generated each time.")
    parser.add_argument("--num_prints", type=int, default=10, help="The number of logging prints per epoch. "
                                                                   "Default: 10")
    args = parser.parse_args()

    args_var = vars(args).copy()
    args_var['csv'] = os.path.splitext(os.path.split(args_var['csv'])[1])[0]
    settings_name = str(args_var).replace("\'", "").replace("{", "").replace("}", "").replace(": ", "=")
    file_handler = logging.FileHandler(f'logs/{settings_name}.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.info(f"====================================================================================================="
                f"\nGrowing Neural Network experiment. "
                f"\nSettings: {args}\n")

    main(args)
