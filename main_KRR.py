import os
import argparse
import time
from utils import *

from dataset import init_dataset
from dataset.gaussian_random_fields import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def kernel_regression(K_trtr, K_tetr, y_tr, y_te, ridge):
    """
    Perform kernel ridge regression
    :param K_trtr: train-train gram matrix
    :param K_tetr: test-train gram matrix
    :param y_tr: training labels
    :param y_te: testing labels
    :param ridge: ridge value
    :return: mean square error.
    """
    alpha = torch.linalg.solve(
        K_trtr + ridge * torch.eye(y_tr.size(0), device=K_trtr.device),
        y_tr
    )

    f = K_tetr @ alpha
    mse = (f - y_te).pow(2).mean()
    return mse


def run_krr(args):

    t1 = time.time()
    def timing_fun(t1):
        t2 = time.time()
        print(print_time(t2 - t1))
        t1 = t2
        return t1

    args.device = device

    # initialize dataset
    print('Init dataset...')
    xtr, ytr, xte, yte = init_dataset(args)
    t1 = timing_fun(t1)

    print('Compute NTK gram matrix (train)...')
    ktrtr = gram_ntk(xtr, xtr)
    t1 = timing_fun(t1)

    print('Compute NTK gram matrix (test)...')
    ktetr = gram_ntk(xte, xtr)
    t1 = timing_fun(t1)

    print('KRR...')
    mse = kernel_regression(ktrtr, ktetr, ytr, yte, args.l)
    timing_fun(t1)

    res = {
        'args': args,
        'mse': mse,
    }

    yield res


def main():
    parser = argparse.ArgumentParser(
        description="Perform KRR on the sphere model"
    )

    """
    	DATASET ARGS
    """
    parser.add_argument("--d", metavar="d", type=int, help="dimension of the input")
    parser.add_argument("--dataseed", type=int, help="dataset seed", default=0)
    parser.add_argument("--pofx", type=str, help="pdf of x", default="normal")
    parser.add_argument("--div", type=float, help="distance spread", default=2.0)

    ### target ###
    parser.add_argument("--target", type=str, help="target function", default="norm")
    parser.add_argument("--teacher_act", type=str, default='abs', help="activation [relu, abs]")
    parser.add_argument("--act_power", type=float, default=2, help="power for the teacher activation")

    """
           TRAINING ARGS
    """
    parser.add_argument("--ptr", metavar="P", type=int, help="size of the training set")
    parser.add_argument("--pte", type=int, help="size of the validation set", default=8192)

    ### ridge parameter ###
    parser.add_argument("--l", metavar="lambda", type=float, help="regularisation parameter")

    parser.add_argument("--pickle", type=str, required=True)

    args = parser.parse_args()

    torch.save(args, args.pickle)
    saved = False

    try:
        for res in run_krr(args):
            with open(args.pickle, "wb") as f:
                torch.save(args, f, _use_new_zipfile_serialization=False)
                torch.save(res, f, _use_new_zipfile_serialization=False)
                saved = True
    except:
        if not saved:
            os.remove(args.pickle)
        raise


if __name__ == "__main__":
    main()
