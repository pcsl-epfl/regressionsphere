import torch

class MSELoss(torch.nn.Module):
    """
    Mean Squared Error loss function: l = 1/P \sum_i (y_i - \hat{y_i}) ^ 2 / (2 * alpha).
    """
    def __init__(self, alpha):
        super(MSELoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, target):
        mse_loss = 0.5 * (output - target) ** 2 / self.alpha ** 1
        return mse_loss.mean()

def regularize(loss, f, args):
    """
    add L1/L2 regularization to the loss.
    :param loss: current loss
    :param f: network function
    :param args: parser arguments
    """
    if args.reg == "l1":
        loss += args.l / args.h * (f.w1.norm(p=2, dim=-1) * f.w2.abs()).sum()
    elif args.reg == "l2":
        for p in f.parameters():
            loss += args.l / (args.h * 2) * p.pow(2).sum()
    else:
        raise ValueError("Regularization must be either `l1` or `l2`!!")
