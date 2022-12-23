import torch
import torch.nn as nn
import torch.nn.functional as F


# Classes
class FocalLoss(nn.CrossEntropyLoss):
    """ Focal loss for classification tasks on imbalanced datasets """

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


class DRLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, label, cur_M, H_length, reg_lam=0.0):
        return dot_loss(output, label, cur_M, H_length, reg_lam)

    
# Functions
def dot_loss(output, label, cur_M, H_length, reg_lam=0.0):
    target = cur_M[:, label].T ## B, d  output: B, d
    # if criterion == 'dot_loss':
    #     loss = - torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1).mean()
    # elif criterion == 'reg_dot_loss':
    dot = torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1)
    with torch.no_grad():
        M_length = torch.sqrt(torch.sum(target ** 2, dim=1, keepdims=False))
    loss = (1/2) * torch.mean(((dot-(M_length * H_length)) ** 2) / (H_length*M_length))
    if reg_lam > 0:
        reg_Eh_l2 = torch.mean(torch.sqrt(torch.sum(output ** 2, dim=1, keepdims=True)))
        loss = loss + reg_Eh_l2*reg_lam
    return loss


def produce_Ew(label, num_classes):
    uni_label, count = torch.unique(label, return_counts=True)
    batch_size = label.size(0)
    uni_label_num = uni_label.size(0)

    assert batch_size == torch.sum(count)
    gamma = batch_size / uni_label_num
    Ew = torch.ones(1, num_classes).cuda(label.device)
    for i in range(uni_label_num):
        label_id = uni_label[i]
        label_count = count[i]
        length = torch.sqrt(gamma / label_count)
        Ew[0, label_id] = length

    return Ew
