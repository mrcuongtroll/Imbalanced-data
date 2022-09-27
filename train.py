import torch
import logging
import argparse
import pandas as pd
import numpy as np
from data_loader.data_loader import make_datasets
from model.GrowingNN import GrowingMLP
from trainer.trainer import Trainer
from utils.utils import test_report


# Logging
logger = logging.getLogger()
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('logs/log.log')
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.NOTSET)


def main(args):
    logger.info("Loading dataset from args.csv...")
    data = pd.read_csv(args.csv)
    try:
        data.pop("Time")
    except:
        pass
    data.head()
    train_loader, dev_loader, test_loader = make_datasets(data=data, label="Class", batch_size=args.batch_size,
                                                          sampling_mode='under-sampling')
    logger.info("... Done!")
    logger.info("Creating model with input_size=29, output_size=2")
    model = GrowingMLP(29, 2, device=args.device)
    logger.info("Initializing parameters states")
    model.init_params_state()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam
    epochs = args.epochs
    lr = args.learning_rate
    logger.info(f"Training GrowingMLP: Criterion={criterion} | Optimizer={optimizer} | Epochs={epochs} | lr={lr}")
    trainer = Trainer(model=model, optimizer=optimizer, learning_rate=args.learning_rate, device=args.device)
    trainer.train(train_loader=train_loader, dev_loader=dev_loader,
                  criterion=criterion, epochs=30*epochs, steps=10, high=0.5, low=0.3)
    train_loader_u, dev_loader_u, test_loader_u = make_datasets(data=data, label="Class", batch_size=args.batch_size,
                                                                sampling_mode=None)
    trainer.train(train_loader=train_loader_u, dev_loader=dev_loader_u,
                  criterion=criterion, epochs=epochs, steps=100, high=0.5, low=0.3)
    test_report(model, test_loader_u)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GrowingNN test")
    parser.add_argument("--csv", type=str, required=True, help="Directory to the csv dataset")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="The learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=100, help='The number of epochs to train')
    parser.add_argument("-b", "--batch_size", type=int, default=64, help='The batch size of split sets')
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Either 'cpu' or 'cuda'")
    args = parser.parse_args()
    logger.info(args)
    main(args)
