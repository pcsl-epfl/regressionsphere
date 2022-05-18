import torch
import math

class MSELoss(torch.nn.Module):
    """
        Mean Squared Error loss function: l = 1/P \sum_i (y_i - \hat{y_i}) ^ 2 / (2 * alpha).
    """
    def __init__(self, alpha):
        super(MSELoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, target):
        if self.alpha != -1:
            mse_loss = 0.5 * (output - target) ** 2 / self.alpha
        else:
            mse_loss = target - output
        return mse_loss.mean()

def regularize(loss, f, l, args):
    """
    add L1/L2 regularization to the loss.
    :param loss: current loss
    :param f: network function
    :param args: parser arguments
    """
    if args.reg == "l1":
        if not args.w1_norm1:
            if f.b:
                bn = f.b.pow(2)
            else:
                bn = 0
            loss += l / args.h * (f.w1.norm(p=2, dim=-1).add(bn) * f.w2.abs()).sum()
        else:
            loss += l / args.h * f.w2.abs().sum()
            if f.b:
                loss += l / args.h * f.b.abs().sum()
    elif args.reg == "l2":
        for p in f.parameters():
            loss += l / (args.h * 2) * p.pow(2).sum()
    else:
        raise ValueError("Regularization must be either `l1` or `l2`!!")

def lambda_decay(args, epoch):
    """
    lambda decay.
    :param args: parser arguments
    :param epoch: current epoch
    """
    if args.l_decay == 'pow_law':
        return 1 / (1 + epoch ** args.l_decay_param)
    elif args.l_decay == 'pl_exp':
        return args.l * math.exp(1 - epoch ** .5 / args.ptr ** .7) / (1 + epoch)
    elif args.l_decay == 'exp':
        return args.l * math.exp(1 - epoch ** args.l_decay_param)
    elif args.l_decay == 'none':
        return args.l
    else:
        raise ValueError("Regularization decay must be `pow_law` or `pl_exp`, `exp` or `none`.")

class Large2zeroLambdaScheduler:
    def __init__(self, l):

        self.l = l
        self.min_loss = 1e10
        self.internal_step = 0
        self.max_internal_step = int(2e6)
        self.time_from_min = 0

    def step(self, loss):
        if self.time_from_min < 10 / self.l:
            if loss >= self.min_loss:
                self.time_from_min += 1
            else:
                self.min_loss = loss
            return self.l, 0
        else:
            self.internal_step += 1
            stop_flag = 0 if self.internal_step < self.max_internal_step else 1
            return 0., stop_flag