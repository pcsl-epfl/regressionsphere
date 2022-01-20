import copy
import time
import torch
import torch.optim as optim

from utils import *
from loss import MSELoss, regularize
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
        print("Epoch : ", epoch + 1, "saving network ...", flush=True)
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

    timeckpt_gen, lossckpt_gen = ckp_init(args, alpha * ltr.detach().item())

    timeckpt = next(timeckpt_gen)
    lossckpt = next(lossckpt_gen)

    start_time = time.time()
    for epoch in range(args.maxstep):

        if torch.isnan(ltr):
            break

        ltr.backward()
        optimizer.step()

        optimizer.zero_grad()

        otr = F(xtr)
        ltr = loss(otr, ytr)
        if args.l:
            regularize(ltr, f, args)

        ltr_val = alpha * ltr.detach().item()

        if ltr_val <= lossckpt:
            save_net()
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

