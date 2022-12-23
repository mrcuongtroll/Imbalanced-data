import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Classes
class _CustomReLU(torch.autograd.Function):

    @staticmethod
    def jvp(ctx, grad_inputs):
        pass

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0.0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input * (x >= 0).float()


class CustomReLU(nn.Module):

    def __init__(self, inplace=False):
        super(CustomReLU, self).__init__()
        self.inplace = inplace  # no use for now

    def forward(self, x):
        return _custom_relu(x)


class ETFClassifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=True, LWS=False, reg_ETF=False, device='cuda'):
        super(ETFClassifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * \
            torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M.to(device)

        # self.LWS = LWS
        # self.reg_ETF = reg_ETF
#        if LWS:
#            self.learned_norm = nn.Parameter(torch.ones(1, num_classes))
#            self.alpha = nn.Parameter(1e-3 * torch.randn(1, num_classes).cuda())
#            self.learned_norm = (F.softmax(self.alpha, dim=-1) * num_classes)
#        else:
#            self.learned_norm = torch.ones(1, num_classes).cuda()

        self.BN_H = nn.BatchNorm1d(feat_in)
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(
            num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        x = self.BN_H(x)
        # x = x @ self.ori_M
        return x


# Functions
def _custom_relu(x):
    return _CustomReLU.apply(x)
