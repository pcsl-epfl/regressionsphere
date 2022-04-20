import torch
import torch.nn.functional as F

class FcTeacher(torch.nn.Module):
    def __init__(self, h, d, h_batch=10000, act='relu', a=1, device='cpu'):
        super(FcTeacher, self).__init__()

        self.d = d
        self.h = int(h)
        self.h_batch = int(h_batch)
        self.a = a
        self.device = device
        if act == 'relu':
            self.act = F.relu
        elif act == 'abs':
            self.act = torch.abs
        else:
            raise ValueError('Only ReLU and abs implemented as teacher activations!!')

    def forward(self, x):

        out = torch.zeros(len(x), device=self.device)

        for i in range(self.h // self.h_batch):
            w1 = torch.randn(self.h_batch, self.d, device=self.device)
            w1 /= w1.norm(dim=-1, keepdim=True)
            o = (x @ w1.t())
            o = self.act(o).pow(self.a)
            out += o @ torch.randn(self.h_batch, device=o.device).sign()

        return out / self.h ** .5
