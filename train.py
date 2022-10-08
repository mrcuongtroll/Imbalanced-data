import sys
import torch
import logging
import argparse
import pandas as pd
import numpy as np
import os
import dill
import torchvision.datasets as datasets
from data_loader.data_loader import make_datasets, make_dataloader, UNDER_SAMPLING, OVER_SAMPLING
from model.GrowingNN import GrowingMLP
from trainer.trainer import Trainer
from utils.utils import test_report, count_parameters


# Constants
CHECKPOINT = './checkpoints'
LOGS = './logs'


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
    args_var = vars(args).copy()
    args_var['dataset'] = os.path.splitext(os.path.split(args_var['dataset'])[1])[0]
    settings_name = str(args_var).replace("\'", "").replace("{", "").replace("}", "").replace(": ", "=")
    logger.info(f"Loading dataset from {args.dataset}...")
    if os.path.isfile(args.dataset) and os.path.splitext(args.dataset)[1] == '.csv':
        data = pd.read_csv(args.dataset)
    # elif args.dataset.lower() == 'mnist':
    #     data =
    if isinstance(data, pd.DataFrame):
        try:
            data.pop("Time")
        except:
            pass
        logger.debug(f"Example samples from the dataset:\n{data.head()}")
    """
    train_loader, dev_loader, test_loader = make_datasets(data=data, label=args.label_col,
                                                          batch_size=args.batch_size,
                                                          sampling_mode=None,
                                                          num_workers=args.num_workers)
    train_loader_r, dev_loader_r, test_loader_r = make_datasets(data=data, label=args.label_col,
                                                                batch_size=args.batch_size,
                                                                sampling_mode=args.resampling_mode,
                                                                num_workers=args.num_workers)
    """
    train_df, test_df = make_datasets(data=data, label=args.label_col,
                                      batch_size=args.batch_size,
                                      splits=(0.9, 0.1),
                                      sampling_mode=None,
                                      return_dframe=True)
    train_df, dev_df = make_datasets(data=train_df, label=args.label_col,
                                     batch_size=args.batch_size,
                                     splits=(0.8, 0.2),
                                     sampling_mode=None,
                                     return_dframe=True)
    train_loader = make_dataloader(data=train_df, label=args.label_col, batch_size=args.batch_size, sampling_mode=None,
                                   num_workers=args.num_workers)
    dev_loader = make_dataloader(data=dev_df, label=args.label_col, batch_size=args.batch_size, sampling_mode=None,
                                 num_workers=args.num_workers)
    train_loader_r = make_dataloader(data=train_df, label=args.label_col, batch_size=args.batch_size,
                                     sampling_mode=args.resampling_mode, num_workers=args.num_workers)
    dev_loader_r = make_dataloader(data=dev_df, label=args.label_col, batch_size=args.batch_size,
                                   sampling_mode=args.resampling_mode, num_workers=args.num_workers)
    test_loader = make_dataloader(data=test_df, label=args.label_col, batch_size=args.batch_size, sampling_mode=None,
                                  num_workers=args.num_workers)
    logger.info("... Done!")
    input_size = len(data.columns) - 1  # Minus the label column
    output_size = len(data[args.label_col].unique())
    logger.info(f"Creating model with input_size={input_size}, output_size={output_size}")
    model = GrowingMLP(input_size, output_size, device=args.device)
    logger.info("Initializing parameters states")
    model.init_params_state()
    if os.path.exists(os.path.join(CHECKPOINT, settings_name, "checkpoint.th")):
        model = torch.load(os.path.join(CHECKPOINT, settings_name, "checkpoint.th"), pickle_module=dill)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam
    epochs = args.epochs
    lr = args.learning_rate
    logger.info(f"Training GrowingMLP: Criterion={criterion} | Optimizer={optimizer} | Epochs={epochs} | lr={lr}")
    trainer = Trainer(model=model, optimizer=optimizer, learning_rate=args.learning_rate, device=args.device,
                      checkpoint_name=settings_name)
    if os.path.exists(os.path.join(CHECKPOINT, settings_name, "trainer.th")):
        trainer_state = torch.load(os.path.join(CHECKPOINT, settings_name, "trainer.th"))
        trainer.optimizer.load_state_dict(trainer_state['optimizer'])
        trainer.train_loss_history = trainer_state['train_loss']
        trainer.dev_loss_history = trainer_state['dev_loss']
    resampling_epochs = 30*epochs if args.resampling_mode in UNDER_SAMPLING else epochs
    params, params_table = count_parameters(model)
    logger.debug(f"Total number of parameters: {params}.\n{params_table}")
    trainer.train(train_loader=train_loader_r, dev_loader=dev_loader_r, gen_rate=args.gen_rate,
                  criterion=criterion, epochs=resampling_epochs, num_prints=args.num_prints,
                  high=args.high, low=args.low)
    trainer.train(train_loader=train_loader, dev_loader=dev_loader, gen_rate=args.gen_rate,
                  criterion=criterion, epochs=epochs, num_prints=args.num_prints, high=args.high, low=args.low)
    params, params_table = count_parameters(model)
    logger.debug(f"Total number of parameters: {params}.\n{params_table}")
    test_report(model, test_loader)
    # logger.debug(f"linear4's weight: {model.linear4.weight}")
    # logger.debug(f"linear4's bias: {model.linear4.bias}")
    # logger.debug(f"linear4's weight mask: {model.params_state['linear4.weight']}")
    # logger.debug(f"linear4's bias mask: {model.params_state['linear4.bias']}")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GrowingNN test")
    parser.add_argument("--dataset", type=str, required=True, help="Directory to the dataset. (Required).")
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
    parser.add_argument("--resampling_mode", type=str, default="under-sampling", help="The resampling mode to create"
                                                                                      " balance set")
    parser.add_argument("-nw", "--num_workers", type=int, default=0, help="Number of workers for dataloaders. 0 means "
                                                                          "that the data will be loaded in the main "
                                                                          "process."
                                                                          "Default: 0")
    args = parser.parse_args()

    args_var = vars(args).copy()
    args_var['dataset'] = os.path.splitext(os.path.split(args_var['dataset'])[1])[0]
    settings_name = str(args_var).replace("\'", "").replace("{", "").replace("}", "").replace(": ", "=")
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file_handler = logging.FileHandler(os.path.join(LOGS, f'{settings_name}.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler(os.path.join(CHECKPOINT, settings_name, "log.log"))
    file_handler2.setFormatter(formatter)
    file_handler2.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.info(f"====================================================================================================="
                f"\nGrowing Neural Network experiment. "
                f"\nSettings: {args}\n")

    main(args)
