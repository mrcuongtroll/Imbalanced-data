import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, classification_report
from prettytable import PrettyTable
import logging


# Logging
logger = logging.getLogger(__name__)


# Functions
def test_report(model: torch.nn.Module, test_loader: DataLoader, ETF=False, device='cuda'):
    y_true = []
    y_pred = []
    model.to(device)
    if ETF:
        softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            if ETF:
                feature = model(data)
                out = softmax(feature @ model.output_layer.ori_M)
            else:
                out = model(data)
            out = torch.argmax(out, dim=1)
            y_pred.append(out)
            y_true.append(target)
    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
    logger.info('\n'+classification_report(y_true, y_pred, digits=4))


def plot_activation(activations, label, save_path=None):
    def plot_layer_activation(layer_activation, ax, layer_name=None):
        if len(layer_activation.shape) == 1:
            ax.bar(range(len(layer_activation)), layer_activation, color='blue')
        elif len(layer_activation.shape) == 2:
            ax.imshow(layer_activation, cmap='Blues', vmin=0, vmax=0.5)
        ax.set_title(layer_name)
        return
    fig, axes = plt.subplots(5, figsize=(20, 20), tight_layout=True)
    fig.suptitle(label)
    if isinstance(activations, dict):
        count = 0
        for layer in activations.keys():
            plot_layer_activation(activations[layer], axes[count], layer_name=f'Layer: {layer}')
            count += 1
    else:
        plot_layer_activation(activations, axes[0])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_pr_curve(model: nn.Module, test_dataset: Dataset, ETF=False, ax=None, title=None, device='cuda'):
    with torch.no_grad():
        if ETF:
            ori_M = model.output_layer.ori_M
            features = model(test_dataset.data.to(device))
            softmax = nn.LogSoftmax(dim=1)
            output = softmax(features @ ori_M)
        else:
            output = model(test_dataset.data.to(device))
        positive_prob = output[:, 1]
    precision, recall, thresholds = precision_recall_curve(test_dataset.label, positive_prob.detach().cpu().numpy())
    area_under_the_curve = auc(recall, precision)
    if ax is None:
        fig, ax = plt.subplots(fig_size=(10, 10))
    line, = ax.plot(recall, precision)
    if title:
        ax.set_title(title, size=20)
        ax.set_ylabel('Precision', size=13)
        ax.set_xlabel('Recall', size=13)
    return line, area_under_the_curve


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    # print(f"Total Trainable Params: {total_params}")
    return total_params, table
