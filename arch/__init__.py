import torch
from torch import nn
from .fibonacci import fibonacci_lattice


class FC(nn.Module):
    """
    Fully connected, one-hidden-layer neural network with ReLU activation.
    """

    def __init__(self, h, d, scale=None, bias=False, device='cpu', w1_init='normal', w2_init='normal'):
        """
        :param h: number of hidden units
        :param d: input space dimension
        :param scale: scale of the output function: `1/h` for the mean-field limit, `1/sqrt(h)` for the NTK limit.
        :param bool bias: first layer bias flag.
        :param device: cpu or cuda.
        :param w1_init: probability distribution for *inner* weights initialization
        :param w2_init: probability distribution for *outer* weights initialization
        """
        super(FC, self).__init__()

        self.d = d
        self.scale = 1 / h if scale is None else scale

        ### Inner weights initialization ###
        assert w1_init in ['normal', 'unitary', 'fibonacci']
        if w1_init == 'fibonacci':
            self.w1 = fibonacci_lattice(d, h).to(device)
        else:
            w1 = torch.randn(h, d, device=device)
            self.w1 = nn.Parameter(w1)
            if w1_init == 'unitary':
                self.project_weight()

        self.b = nn.Parameter(torch.zeros(h, device=device)) if bias else 0

        ### Outer weights initialization ###
        if w2_init == 'normal':
            self.w2 = nn.Parameter(torch.randn(1, h, device=device))
        elif w2_init == 'zero':
            self.w2 = nn.Parameter(torch.zeros(1, h, device=device))
        elif w2_init == 'one':
            self.w2 = nn.Parameter(torch.randn(1, h, device=device).sign())
        elif is_number(w2_init):
            self.w2 = nn.Parameter(torch.randn(1, h, device=device) * float(w2_init))
        else:
            raise ValueError('Weights initialization must be either `normal` or `zero` or float!')

    def forward(self, x):
        x = x @ self.w1.t() + self.b
        x = x.relu()
        x = x @ self.w2.t()
        return x.squeeze() * self.scale

    def project_weight(self):
        with torch.no_grad():
            self.w1 /= self.w1.norm(dim=-1, keepdim=True)

    def project_grad(self):
        self.w1.grad -= (self.w1.grad * self.w1).sum(dim=-1, keepdim=True) * self.w1

    def conic_gd(self):
        self.w1.grad /= self.w2.t().pow(self.pow)
        if self.pow == 1:
            self.w2.grad *= self.w2


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
