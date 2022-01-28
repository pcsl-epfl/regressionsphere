import torch
from torch import nn
import torch.nn.functional as F
from .fibonacci import uniform_hypersphere

class FC(nn.Module):
    """
    Fully connected, one-hidden-layer neural network with ReLU activation.
    """
    def __init__(self, h, d, scale=None, bias=False, device='cpu', w2_init='normal', fibonacci=False):
        """
        :param h: number of hidden units
        :param d: input space dimension
        :param scale: scale of the output function: `1/h` for the mean-field limit, `1/sqrt(h)` for the NTK limit.
        :param bool bias: first layer bias flag.
        :param device: cpu or cuda.
        """
        super(FC, self).__init__()

        self.d = d
        if scale is None:
            self.scale = 1 / h
        else:
            self.scale = scale
        if fibonacci:
            self.w1 = uniform_hypersphere(d, h).to(device)
        else:
            self.w1 = nn.Parameter(torch.randn(h, d, device=device))

        if bias:
            self.b1 = nn.Parameter(torch.randn(h))
        else:
            self.register_parameter("b1", None)
        if w2_init == 'normal':
            self.w2 = nn.Parameter(torch.randn(1, h, device=device).abs())
        elif w2_init == 'zero':
            self.w2 = nn.Parameter(torch.zeros(1, h, device=device).abs())
        else:
            raise ValueError('Weights initialization must be either `normal` or `zero`!')

    def forward(self, x):
        x = F.linear(x, self.w1, bias=self.b1) / self.d ** .5
        x = F.relu(x)
        x = F.linear(x, self.w2, bias=None)
        return x.squeeze() * self.scale

