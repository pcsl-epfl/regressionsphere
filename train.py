import copy
import time
import numpy as np
import torch
import torch.optim as optim

from utils import *
from loss import MSELoss, regularize, lambda_decay
from dataset import init_dataset
from arch import FC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

def run_training(args):
    """
    train a fcnn with (full-batch) gradient descent on the mse loss.
    :param args: parser arguments.
    :return: results dictionary.
    """

    args.device = device

    # initialize dataset
    xtr, ytr, xte, yte = init_dataset(args)

    # initialize network function
    torch.manual_seed(args.netseed)
    f = FC(args.h, args.d, bias=args.bias, w1_init=args.init_w1, w2_init=args.init_w2, device=device)
    if not args.train_w1:
        for param in [p for p in f.parameters()][:-1]:
            param.requires_grad = False
    f0 = copy.deepcopy(f)

    # initialize loss function
    alpha = args.alpha
    loss = MSELoss(alpha=alpha)

    # define optimizer
    optimizer = optim.SGD(f.parameters(), lr=args.lr * args.h, weight_decay=0)
    optimizer.zero_grad()
    if args.w1_norm1:
        f.project_weight()

    # define predictor
    def F(x):
        if args.minus_f0:
            return alpha * (f(x) - f0(x))
        else:
            return alpha * f(x)

    # save the network
    def save_net():
        with torch.no_grad():
            ote = F(xte)
        lte = alpha * loss(ote, yte).item()
        state = {
            "t": epoch + 1,
            "train": ltr_val,
            "test": lte,
            "f": copy.deepcopy(f.state_dict()),
        }
        print("Epoch : ", epoch + 1, "saving network ...", flush=True)
        dynamics_state.append(state)

    def count_atoms():
        """TODO: obsolete, have new implementation with hashing"""
        if args.count_atoms:
            w1 = f.w1.detach() * args.alpha ** .5
            w2 = f.w2.detach() * args.alpha ** .5
            norm = (w1.norm(dim=-1) * w2.abs())
            hist = torch.histogram(torch.atan2(*w1.t()[:2]).cpu(), weight=norm[0].cpu(), bins=torch.linspace(0, math.pi, 512), density=True).hist
            a = np.unique((w1[norm[0] > .01] @ xtr.t()).sign().cpu(), axis=0, return_counts=True)[1]
            return sum(a > 1), hist
        else:
            return None, None

    otr = F(xtr)
    ltr = loss(otr, ytr)
    regularize(ltr, f, 0.5, args)

    with torch.no_grad():
        ote = F(xte)
    lte = alpha * loss(ote, yte)

    dynamics_loss = [[0, alpha * ltr.detach().item(), lte.item()]]

    dynamics_state = [
        {
            "t": 0,
            "train": ltr.detach().item(),
            "test": lte.item(),
            "f": copy.deepcopy(f.state_dict()),
        }
    ]

    dynamics_atoms = []

    timeckpt_gen, lossckpt_gen = ckp_init(args, alpha * ltr.detach().item())

    timeckpt = next(timeckpt_gen)
    lossckpt = next(lossckpt_gen)

    start_time = time.time()
    stop_flag = 0
    for epoch in range(args.maxstep):

        if epoch == args.lr_decay_epoch:
            print('lr decay..!')
            optimizer.param_groups[0]['lr'] /= 10

        if stop_flag:
            print("Stopping flag is True!")
            break

        if torch.isnan(ltr):
            print('Train loss is NaN!!')
            break

        ltr.backward()
        if args.conic_gd and (epoch > 5 or args.init_w2 != 'zero'):
            f.conic_gd()
        if args.w1_norm1:
            f.project_grad()
        optimizer.step()

        optimizer.zero_grad()
        if args.w1_norm1:
            f.project_weight()

        otr = F(xtr)
        ltr = loss(otr, ytr)

        ltr_val = alpha * ltr.detach().item()
        if ltr_val < 1e-30 and args.alpha > 0: stop_flag = True

        if args.l:
            l = lambda_decay(args, epoch)
            regularize(ltr, f, l, args)

        if ltr_val <= lossckpt or epoch % (args.maxstep // 50) == 0:
            print('LOSS CKP: saving net...')
            save_net()
            if args.count_atoms:
                na, hist = count_atoms()
                dynamics_atoms.append({"N_A": na, "hist": hist})
            lossckpt = next(lossckpt_gen)

        if (epoch + 1) == timeckpt:
            with torch.no_grad():
                ote = F(xte)
            lte = alpha * loss(ote, yte).item()
            avg_epoch_time = (time.time() - start_time) / (epoch + 1)
            print(f"[Epoch : {int(epoch+1)} / {int(args.maxstep)}, "
                  f"ETA: {print_time(avg_epoch_time * (args.maxstep - epoch - 1))}] "
                  f"\t tr_loss: {ltr_val:.02e}, \t te_loss: {lte:.02e}",
                  flush=True)
            dynamics_loss.append([epoch + 1, ltr_val, lte])
            if args.count_atoms:
                na, hist = count_atoms()
                dynamics_atoms.append({"N_A": na, "hist": hist})
            lossckpt = next(lossckpt_gen)
            timeckpt = next(timeckpt_gen)

    if dynamics_state[len(dynamics_state) - 1]["t"] != args.maxstep:
        save_net()

    f_info = {
        "d": args.d,
        "seed": args.netseed,
        "init": copy.deepcopy(f0.state_dict()),
        "alpha": alpha,
        "dynamics": dynamics_state,
    }

    learning_params = {
        "epoch": epoch + 1,
        "lr": args.lr,
        "reg": args.reg,
        "lambda": args.l,
        "lckpt": lossckpt,
    }

    dataset = {
        'xtr': xtr,
        'ytr': ytr,
        'xte': xte,
        'yte': yte,
    }

    res = {
        "args": args,
        "dataset": dataset,
        "dynamics": dynamics_loss,
        "atoms": dynamics_atoms,
        "f": f_info,
        "learn": learning_params}

    yield res
