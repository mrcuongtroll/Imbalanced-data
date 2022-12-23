import sys
import torch
import logging
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import json
import dill
import torchvision.datasets as datasets
from data_loader.data_loader import *
from model.GrowingNN import GrowingMLP, GrowingCNN
from model import losses
from trainer.trainer import Trainer
from utils.utils import test_report, count_parameters, plot_activation
from definitions import *


# Logging
if not os.path.exists(os.path.abspath(LOGS_DIR)):
    os.makedirs(os.path.abspath(LOGS_DIR))
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
    # Get dataset statistics
    with open(os.path.join(DATASETS_METADATA_DIR, f'{args.dataset.lower()}.json'), 'r') as f:
        dataset_metadata = json.load(f)
    if dataset_metadata['type'] == 'csv':
        data = pd.read_csv(os.path.join(DATASETS_DIR, f'{args.dataset}.csv'))
        model_class = GrowingMLP
        for col in dataset_metadata['pop_cols']:
            data.pop(col)
        label = dataset_metadata['label_col']
    elif dataset_metadata['type'] == 'torchvisionDataset':
        data = DATASETS_MAP[args.dataset.lower()]
        model_class = ARCHITECTURES_MAP[args.architecture.lower()]
        # model_class = GrowingMLP
    else:
        logger.critical("Invalid argument I guess...")
        data = None
        model_class = None
        exit()
    if isinstance(data, pd.DataFrame):
        logger.debug(f"Example samples from the dataset:\n{data.head()}")
    """
    train_loader, dev_loader, test_loader = make_df_datasets(data=data, label=args.label_col,
                                                             batch_size=args.batch_size,
                                                             sampling_mode=None,
                                                             num_workers=args.num_workers)
    train_loader_r, dev_loader_r, test_loader_r = make_df_datasets(data=data, label=args.label_col,
                                                                   batch_size=args.batch_size,
                                                                   sampling_mode=args.resampling_mode,
                                                                   num_workers=args.num_workers)
    """
    if dataset_metadata['type'] == 'csv':
        train_df, test_df = make_df_datasets(data=data, label=label,
                                             batch_size=args.batch_size,
                                             splits=(0.9, 0.1),
                                             sampling_mode=None,
                                             return_dframe=True)
        train_df, dev_df = make_df_datasets(data=train_df, label=label,
                                            batch_size=args.batch_size,
                                            splits=(0.8, 0.2),
                                            sampling_mode=None,
                                            return_dframe=True)
        train_loader = make_df_dataloader(data=train_df, label=label, batch_size=args.batch_size, sampling_mode=None,
                                          num_workers=args.num_workers)
        dev_loader = make_df_dataloader(data=dev_df, label=label, batch_size=args.batch_size, sampling_mode=None,
                                        num_workers=args.num_workers)
        train_loader_r = make_df_dataloader(data=train_df, label=label, batch_size=args.batch_size,
                                            sampling_mode=args.resampling_mode, num_workers=args.num_workers)
        dev_loader_r = make_df_dataloader(data=dev_df, label=label, batch_size=args.batch_size,
                                          sampling_mode=args.resampling_mode, num_workers=args.num_workers)
        test_loader = make_df_dataloader(data=test_df, label=label, batch_size=args.batch_size, sampling_mode=None,
                                         num_workers=args.num_workers)
    elif dataset_metadata['type'] == 'torchvisionDataset':
        train_loader_r, test_loader_r = make_tv_dataloaders(
            dataset_name=args.dataset.lower(),
            batch_size=args.batch_size,
            flatten_images=FLATTEN_FOR_ARCH[args.architecture.lower()]
        )
        dev_loader_r = test_loader_r
        train_loader = train_loader_r
        dev_loader = dev_loader_r
        test_loader = test_loader_r
    else:
        train_loader, dev_loader, test_loader = None, None, None
        logger.critical("Invalid argument bruh")
        exit()
    logger.info("... Done!")
    # input_size = len(data.columns) - 1  # Minus the label column
    # output_size = len(data[args.label_col].unique())
    model_args = dataset_metadata['model_args']
    logger.info(f"Creating model with input_size={model_args['input_size']}, output_size={model_args['output_size']}")
    if args.proposed_method:
        # model = model_class(input_size, output_size, device=args.device, growing_method='random')
        if args.ETF:
            logger.info("Model is using an ETF classifier")
            model = model_class(**model_args, ETF=True, device=args.device, growing_method='gradmax')
        else:
            model = model_class(**model_args, ETF=False, device=args.device, growing_method='gradmax')
        logger.info("Initializing parameters states")
        model.init_params_state()
    else:
        logger.info("Not applying proposed method. Initializing networks with higher params count.")
        # model = model_class(**model_args, hidden_sizes=(198, 394, 789, 394, 198))
        model = model_class(**model_args, ETF=False)
    if os.path.exists(os.path.join(CHECKPOINT_DIR, settings_name, "checkpoint.th")):
        model = torch.load(os.path.join(CHECKPOINT_DIR, settings_name, "checkpoint.th"), pickle_module=dill)
    if args.DRLoss:
        criterion = losses.DRLoss()
    else:
        criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam
    epochs = args.epochs
    lr = args.learning_rate
    logger.info(f"Training GrowingMLP: Criterion={criterion} | Optimizer={optimizer} | Epochs={epochs} | lr={lr}")
    trainer = Trainer(model=model, optimizer=optimizer, ETF=args.ETF, learning_rate=args.learning_rate, device=args.device,
                      checkpoint_name=settings_name)
    if os.path.exists(os.path.join(CHECKPOINT_DIR, settings_name, "trainer.th")):
        trainer_state = torch.load(os.path.join(CHECKPOINT_DIR, settings_name, "trainer.th"))
        trainer.optimizer.load_state_dict(trainer_state['optimizer'])
        trainer.train_loss_history = trainer_state['train_loss']
        trainer.dev_loss_history = trainer_state['dev_loss']
    resampling_epochs = 5*epochs if args.resampling_mode in UNDER_SAMPLING else epochs
    params, params_table = count_parameters(model)
    logger.debug(f"Total number of parameters: {params}.\n{params_table}")
    if args.proposed_method:
        trainer.train(train_loader=train_loader_r, dev_loader=dev_loader_r, gen_rate=args.gen_rate,
                      criterion=criterion, epochs=resampling_epochs, num_prints=args.num_prints,
                      high=args.high, low=args.low)
        # trainer.train(train_loader=train_loader, dev_loader=dev_loader, gen_rate=args.gen_rate,
        #               criterion=criterion, epochs=epochs, num_prints=args.num_prints, high=args.high, low=args.low)
    else:
        # trainer.train(train_loader=train_loader_r, dev_loader=dev_loader_r, gen_rate=args.gen_rate,
        #               criterion=criterion, epochs=resampling_epochs, num_prints=args.num_prints,
        #               high=args.high, low=args.low, freeze_neurons=False)
        trainer.train(train_loader=train_loader, dev_loader=dev_loader, gen_rate=args.gen_rate,
                      criterion=criterion, epochs=epochs, num_prints=args.num_prints, high=args.high, low=args.low,
                      freeze_neurons=False)
    params, params_table = count_parameters(model)
    logger.debug(f"Total number of parameters: {params}.\n{params_table}")
    test_report(model, test_loader, ETF=args.ETF)

    # Activation plot
    with torch.no_grad():
        test_dataset = test_loader.dataset
        samples = []
        target = []
        count = 0
        target_label = 0
        for sample, label in test_dataset:
            # if target_label == len(dataset_metadata['classes']):
            if target_label == 2:
                break
            if label == target_label:
                if args.ETF:
                    features = model(sample.unsqueeze(0).to(args.device))
                    softmax = nn.Softmax()
                else:
                    out = model(sample.unsqueeze(0).to(args.device))
                    pred = out.argmax(dim=1)
                if pred.item() == label.item(): # and out.exp().max() >= 0.95:
                    samples.append(sample.detach().cpu().numpy())
                    target.append(out.exp()[0, 1].item())
                    count += 1
            if count == 10:
                target_label += 1
                count = 0
        samples = torch.tensor(np.array(samples), dtype=torch.float)
        target = torch.tensor(np.array(target), dtype=torch.float)
        # data, target = next(iter(test_loader))
        samples, target = samples.to(args.device), target.to(args.device)
        pred = model(samples)
        target = np.around(target.detach().cpu().numpy(), decimals=3)
        if isinstance(model, GrowingCNN):
            plot_activation(model.nonzero_pct, target,
                            save_path=os.path.join(CHECKPOINT_DIR, settings_name, 'activation.png'))
        else:
            plot_activation(model.activation_table, target,
                            save_path=os.path.join(CHECKPOINT_DIR, settings_name, 'activation.png'))
        del data, target, pred
    # logger.debug(f"linear4's weight: {model.linear4.weight}")
    # logger.debug(f"linear4's bias: {model.linear4.bias}")
    # logger.debug(f"linear4's weight mask: {model.params_state['linear4.weight']}")
    # logger.debug(f"linear4's bias mask: {model.params_state['linear4.bias']}")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GrowingNN test")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name. (Required).")
    parser.add_argument("--architecture", type=str, required=True, help="Architecture name. (Required).")
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
    parser.add_argument("--proposed_method", default=True, action='store_true',
                        help="To apply the proposed method.")
    parser.add_argument("--no_proposed_method", dest="proposed_method", action='store_false',
                        help="Do not apply the proposed method.")
    parser.add_argument("--ETF", action="store_true", default=False, help="To apply the ETF Classifier")
    parser.add_argument("--DRLoss", action='store_true', default=False, help="To apply the Dot Regression Loss")
    args = parser.parse_args()

    args_var = vars(args).copy()
    if os.path.isfile(args_var['dataset']):
        args_var['dataset'] = os.path.splitext(os.path.split(args_var['dataset'])[1])[0]
    settings_name = str(args_var).replace("\'", "").replace("{", "").replace("}", "").replace(": ", "=")
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    file_handler = logging.FileHandler(os.path.join(LOGS_DIR, f'{settings_name}.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    if not os.path.exists(os.path.join(CHECKPOINT_DIR, settings_name)):
        os.makedirs(os.path.join(CHECKPOINT_DIR, settings_name))
    file_handler2 = logging.FileHandler(os.path.join(CHECKPOINT_DIR, settings_name, "log.log"))
    file_handler2.setFormatter(formatter)
    file_handler2.setLevel(logging.DEBUG)
    logger.addHandler(file_handler2)
    logger.addHandler(file_handler)
    logger.info(f"====================================================================================================="
                f"\nGrowing Neural Network experiment. "
                f"\nSettings: {args}\n")

    main(args)
