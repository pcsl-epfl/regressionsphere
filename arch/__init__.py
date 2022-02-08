import torch
from torch import nn
import torch.nn.functional as F
from .fibonacci import fibonacci_lattice

class FC(nn.Module):
    """
    Fully connected, one-hidden-layer neural network with ReLU activation.
    """
    def __init__(self, h, d, scale=None, bias=False, device='cpu', w1_init='normal', w2_init='normal', w1_onsphere=False):
        """
        :param h: number of hidden units
        :param d: input space dimension
        :param scale: scale of the output function: `1/h` for the mean-field limit, `1/sqrt(h)` for the NTK limit.
        :param bool bias: first layer bias flag.
        :param device: cpu or cuda.
        """
        super(FC, self).__init__()

        self.d = d
        self.w1_onsphere = w1_onsphere
        if scale is None:
            self.scale = 1 / h
        else:
            self.scale = scale
        assert w1_init in ['normal', 'unitary', 'fibonacci', 'small']
        if w1_init == 'fibonacci':
            self.w1 = fibonacci_lattice(d, h).to(device)
        else:
            w1 = torch.randn(h, d, device=device)
            if w1_init == 'small':
                w1 *= 1e-50
            if w1_init == 'unitary':
                w1 = w1 / w1.norm(dim=-1, keepdim=True)

            self.w1 = nn.Parameter(w1)

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
        if self.w1_onsphere:
            w1 = self.w1 / self.w1.norm(dim=-1, keepdim=True)
            scale1 = 1
        else:
            w1 = self.w1
        scale1 = self.d ** .5
        x = F.linear(x, w1, bias=self.b1) / scale1
        x = F.relu(x)
        x = F.linear(x, self.w2, bias=None)
        return x.squeeze() * self.scale

