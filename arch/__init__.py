import torch
from torch import nn
import torch.nn.functional as F

class FC(nn.Module):
    """
    Fully connected, one-hidden-layer neural network with ReLU activation.
    """
    def __init__(self, h, d, scale=None, bias=False, device='cpu'):
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
        self.w1 = nn.Parameter(torch.randn(h, d, device=device))
        if bias:
            self.b1 = nn.Parameter(torch.randn(h))
        else:
            self.register_parameter("b1", None)
        self.w2 = nn.Parameter(torch.randn(1, h, device=device).abs())

    def forward(self, x):
        x = F.linear(x, self.w1, bias=self.b1) / self.d ** .5
        x = F.relu(x)
        x = F.linear(x, self.w2, bias=None)
        return x.squeeze() * self.scale


# class FC(nn.Module):
#     """
#     Fully connected, one-hidden-layer neural network with ReLU activation.
#     """
#     def __init__(self, h, d, scale=None, bias=False, device='cpu', param_list=False):
#         """
#         :param h: number of hidden units
#         :param d: input space dimension
#         :param scale: scale of the output function: `1/h` for the mean-field limit, `1/sqrt(h)` for the NTK limit.
#         :param bool bias: first layer bias flag.
#         :param device: cpu or cuda.
#         """
#         super(FC, self).__init__()
#
#         self.d = d
#         if scale is None:
#             self.scale = 1 / h
#         else:
#             self.scale = scale
#         w1 = torch.randn(h, d, device=device)
#         if param_list:
#             n = max(1, 128 * 256 // h)
#             self.w1 = nn.ParameterList([nn.Parameter(w1[j: j + n]) for j in range(0, h, n)])
#         else:
#             self.w1 = nn.Parameter(w1)
#         if bias:
#             self.b1 = nn.Parameter(torch.randn(h))
#         else:
#             self.register_parameter("b1", None)
#
#         w2 = torch.randn(1, h, device=device)
#         if param_list:
#             self.w2 = nn.ParameterList([nn.Parameter(w2[:, j: j + n]) for j in range(0, h, n)])
#         else:
#             self.w2 = nn.Parameter(w2)
#
#     def forward(self, x):
#         if isinstance(self.w1, nn.ParameterList):
#             w1 = torch.cat(list(self.w1))
#             w2 = torch.cat(list(self.w2))
#         x = F.linear(x, w1, bias=self.b1) / self.d ** .5
#         x = F.relu(x)
#         x = F.linear(x, w2, bias=None)
#         return x.squeeze() * self.scale