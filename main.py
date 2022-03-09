import os
import argparse

import torch
from train import run_training

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
    parser.add_argument("--teacher_act", type=str, default='abs', help="activation [relu, abs]")
    parser.add_argument("--act_power", type=float, default=2, help="power for the teacher activation")

    """
    	ARCHITECTURE ARGS
    """
    parser.add_argument("--h", metavar="H", type=int, help="width of the f network")
    parser.add_argument("--netseed", type=int, help="seed for the network", default=-1)
    parser.add_argument("--alpha", type=float, metavar="alpha", default=1.)
    parser.add_argument("--minus_f0", type=int, default=1)
    parser.add_argument("--lr", type=float, metavar="lr", help="lr", default=1.0)
    parser.add_argument("--bias", type=int, default=0, help="bias")
    parser.add_argument("--init_w1", type=str, default='normal', help="first layer weights initialization")
    parser.add_argument("--init_w2", type=str, default='normal', help="second layer weights initialization")
    parser.add_argument("--train_w1", type=int, default=1, help="train first layer weights")
    parser.add_argument("--w1_norm1", type=int, default=0, help="constrain w1 on the sphere")
    """
           TRAINING ARGS
    """
    parser.add_argument("--ptr", metavar="P", type=int, help="size of the training set")
    parser.add_argument("--pte", type=int, help="size of the validation set", default=8192)
    parser.add_argument("--reg", type=str, help="l1,l2", default="l2")
    parser.add_argument("--l", metavar="lambda", type=float, help="regularisation parameter")
    parser.add_argument("--l_decay", type=str, default='none', help="lr decay")
    parser.add_argument("--l_decay_param", type=float, default=2, help="lr decay parameter")
    parser.add_argument("--count_atoms", type=int, default=0, help="count the number of atoms")
    """
    	OUTPUT ARGS
    """
    # parser.add_argument("--maxtime", type=float, help="maximum time in hours", default=2)
    parser.add_argument("--maxstep", type=float, help="maximum amount of steps of GD", default=20000)
    parser.add_argument("--savefreq", type=int, help="frequency of saves in steps", default=1000)

    parser.add_argument("--pickle", type=str, required=True)

    args = parser.parse_args()

    if args.netseed == -1:
        args.netseed = args.dataseed
    if args.pte == -1:
        args.pte = args.ptr * 4
    args.maxstep = int(args.maxstep)
    # args.l_init = args.l
    # if args.l_final != -1:
    #     args.l_step = (args.l_final / args.l) ** (1 / args.maxstep)
    # else:
    #     args.l_step = 1

    torch.save(args, args.pickle)
    saved = False
    try:
        for res in run_training(args):
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
