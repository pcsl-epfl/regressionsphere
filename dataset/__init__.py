import torch

def init_dataset(args):
    """
    Initialize the sphere dataset for regression.
    Data-points are sampled from a Gaussian or uniform pdf in the d-sphere.
    The target function is the data-points norm.
    :param args: parser arguments.
    :return: (trainset samples, trainset targets, testset samples, testset targets).
    """

    d = args.d
    ptr = args.ptr
    pte = args.pte
    p = ptr + pte

    torch.manual_seed(args.dataseed)

    if args.pofx == 'normal':
        x = torch.randn(p, d, device=args.device)
    elif args.pofx == 'uniform':
        u = torch.randn(p, d, device=args.device)
        norm = u.norm(dim=1, keepdim=True)
        r = torch.rand(p, 1).pow(1 / d)
        x = r * u / norm
    else:
        raise NotImplementedError

    if args.target == 'norm':
        target = torch.norm(x, dim=1)
    else:
        raise NotImplementedError

    xtr = x[:ptr]
    ytr = target[:ptr]
    xte = x[ptr:]
    yte = target[ptr:]

    return xtr, ytr, xte, yte