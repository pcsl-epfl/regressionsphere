import os
import argparse

import torch
from train import train

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
        for res in train(args):
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
