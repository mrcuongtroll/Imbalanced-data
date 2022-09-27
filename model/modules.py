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


# Functions
def _custom_relu(x):
    return _CustomReLU.apply(x)
