import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, classification_report
import logging


# Logging
logger = logging.getLogger(__name__)


# Functions
def test_report(model: torch.nn.Module, test_loader: DataLoader, device='cuda'):
    y_true = []
    y_pred = []
    model.to(device)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            out = torch.argmax(out, dim=1)
            y_pred.append(out)
            y_true.append(target)
    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
    logger.info('\n'+classification_report(y_true, y_pred, digits=4))


def plot_activation(activations, label):
    def plot_layer_activation(layer_activation, ax, layer_name=None):
        if len(layer_activation.shape) == 1:
            ax.bar(range(len(layer_activation)), layer_activation, color='blue')
        elif len(layer_activation.shape) == 2:
            ax.imshow(layer_activation, cmap='Blues')
        ax.set_title(layer_name)
        return
    fig, axes = plt.subplots(5, figsize=(20,20), tight_layout=True)
    fig.suptitle(label)
    if isinstance(activations, dict):
        for layer in activations.keys():
            plot_layer_activation(activations[layer], axes[layer], layer_name=f'Layer: {layer}')
    else:
        plot_layer_activation(activations, axes[0])


def plot_pr_curve(model: nn.Module, test_dataset: Dataset, ax=None, title=None, device='cuda'):
    with torch.no_grad():
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
