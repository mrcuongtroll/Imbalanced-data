import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class MyDataset(Dataset):
    def __init__(self, data, label):
        super(MyDataset, self).__init__()
        self.data = torch.FloatTensor(data.values)
        # if len(self.sents.shape) == 2:
        #   self.sents = self.sents.reshape((self.sents.shape[0],
        #                                    self.sents.shape[1],
        #                                    1))
        self.label = torch.tensor(label.values, dtype=torch.long)

    def __len__(self):
        return len(self.label)
  
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def plot_percentage(self, title='Percentage of data classes', class_names=None, ax=None):
        labels = self.label.detach().cpu().numpy()
        classes = np.sort(np.unique(labels))
        class_count = []
        class_percentage = []
        for c in classes:
            count = np.sum(labels == c)
            class_count.append(count)
            class_percentage.append(100*count/len(labels))
        if ax is None:
            fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(7, 7))
        if class_names is None:
            class_names = classes
        ax.set_title(title, size=20)
        ax.bar(class_names, class_percentage)
        count_iter = iter(class_count)
        for bar in ax.patches:
            ax.annotate(str(next(count_iter)),
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='center', size=15, xytext=(0, 8), textcoords='offset points')
        return class_count


# Functions
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
