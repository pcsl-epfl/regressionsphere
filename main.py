import os
import copy
import argparse

import torch
import torch.optim as optim

from loss import MSELoss, regularize
from dataset import init_dataset
from arch import FC

import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

def execute(args):

    args.device = device

    # initialize dataset
    xtr, ytr, xte, yte = init_dataset(args)

    # initialize network function
    torch.manual_seed(args.netseed)
    f = FC(args.h, args.d, bias=args.bias).to(device)
    f0 = copy.deepcopy(f)

    # initialize loss function
    alpha = args.alpha
    loss = MSELoss(alpha=alpha)

    # define optimizer
    optimizer = optim.SGD(f.parameters(), lr=args.lr * args.h, weight_decay=0)
    optimizer.zero_grad()

    # define predictor
    def F(x):
        return alpha * (f(x) - f0(x))

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
        print("Epoch : ", epoch + 1, "saving network ...")
        dynamics_state.append(state)

    otr = F(xtr)
    ltr = loss(otr, ytr)
    regularize(ltr, f, args)

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

    timeckpt_gen, lossckpt_gen = ckp_init(args, ltr.detach().item())

    timeckpt = next(timeckpt_gen)
    lossckpt = next(lossckpt_gen)

    for epoch in range(args.maxstep):

        ltr.backward()
        optimizer.step()

        optimizer.zero_grad()

        otr = F(xtr)
        ltr = loss(otr, ytr)
        regularize(ltr, f, args)

        ltr_val = alpha * ltr.detach().item()

        if ltr_val <= lossckpt:
            save_net()
            lossckpt = next(lossckpt_gen)

        if (epoch + 1) == timeckpt:
            with torch.no_grad():
                ote = F(xte)
            lte = alpha * loss(ote, yte).item()
            print(f"Epoch : {int(epoch+1)} / {int(args.maxstep)} \t tr_loss: {ltr_val:.02e}, \t te_loss: {lte:.02e}")
            dynamics_loss.append([epoch + 1, ltr_val, lte])
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

    res = {"args": args, "dynamics": dynamics_loss, "f": f_info, "learn": learning_params}

    yield res


def main():
    parser = argparse.ArgumentParser(
        description="Train a FCNN on the sphere model"
    )

    """
    	DATASET ARGS
    """
    parser.add_argument("--d", metavar="d", type=int, help="dimension of the input")
    parser.add_argument("--dataseed", type=int, help="dataset seed", default=0)
    parser.add_argument("--pofx", type=str, help="pdf of x", default="normal")
    parser.add_argument("--target", type=str, help="target function", default="norm")

    """
    	ARCHITECTURE ARGS
    """
    parser.add_argument("--h", metavar="H", type=int, help="width of the f network")
    parser.add_argument("--netseed", type=int, help="seed for the network")
    parser.add_argument("--alpha", type=float, metavar="alpha", default=1e0)
    parser.add_argument("--lr", type=float, metavar="lr", help="lr", default=1.0)
    parser.add_argument("--bias", type=int, default=0, help="bias")
    """
           TRAINING ARGS
    """
    parser.add_argument("--ptr", metavar="P", type=int, help="size of the training set")
    parser.add_argument("--pte", type=int, help="size of the validation set", default=2048)
    parser.add_argument("--reg", type=str, help="l1,l2", default="l2")
    parser.add_argument("--l", metavar="lambda", type=float, help="regularisation parameter")
    """
    	OUTPUT ARGS
    """
    parser.add_argument("--maxtime", type=float, help="maximum time in hours", default=2)
    parser.add_argument("--maxstep", type=int, help="maximum amount of steps of GD", default=20000)
    parser.add_argument("--savefreq", type=int, help="frequency of saves in steps", default=1000)

    parser.add_argument("--pickle", type=str, required=True)

    args = parser.parse_args()
    torch.save(args, args.pickle)
    saved = False
    try:
        for res in execute(args):
            with open(args.pickle, "wb") as f:
                torch.save(args, f, _use_new_zipfile_serialization=False)
                torch.save(res, f, _use_new_zipfile_serialization=False)
                saved = True
    except:
        if not saved:
            os.remove(args.pickle)
        raise


def loss_checkpoint(init, end=None):
    l = init
    while end is None:
        yield l
        l /= math.sqrt(10.0)

def ckp_init(args, init_loss):

    freq = args.savefreq
    max_step = int(args.maxstep)

    step = freq
    space = step ** (1.0 / 10)
    start = 1.0
    checkpoints = []
    for i in range(9):
        start *= space
        checkpoints.append(int(start))
    while step <= 10 * max_step:
        checkpoints.append(step)
        step += freq

    timeckpt_gen = iter(checkpoints)
    lossckpt_gen = loss_checkpoint(args.alpha * init_loss)

    return timeckpt_gen, lossckpt_gen

if __name__ == "__main__":
    main()
