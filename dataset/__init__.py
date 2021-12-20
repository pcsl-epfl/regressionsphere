import torch

def init_dataset(args):

    d = args.d
    ptr = args.ptr
    pte = args.pte

    torch.manual_seed(args.dataseed)

    if args.pofx == 'normal':
        x = torch.randn(ptr + pte, d, device=args.device)
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