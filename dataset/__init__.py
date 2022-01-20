import torchvision
from .utils import *

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

    transform = torchvision.transforms.ToTensor()

    torch.manual_seed(args.dataseed)

    target = None

    if args.pofx == 'normal':
        x = torch.randn(p, d, device=args.device)
    elif args.pofx == 'uniform':
        u = torch.randn(p, d, device=args.device)
        norm = u.norm(dim=1, keepdim=True)
        r = torch.rand(p, 1).pow(1 / d)
        x = r * u / norm
    elif args.pofx == "mnist_pca":
        tr = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=True, download=True, transform=transform)
        te = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=False, transform=transform)
        x, y, i = intertwine_labels(*dataset_to_tensors(list(tr) + list(te)))
        x = center_normalize(x)
        x = pca(x, d, whitening=True)
        x, target, _ = intertwine_split(x, y, i, [p], [args.dataseed], y.unique())[0]
    else:
        raise NotImplementedError

    if target is None:
        if args.target == 'norm':
            target = torch.norm(x, dim=1)
        else:
            raise NotImplementedError

    xtr = x[:ptr]
    ytr = target[:ptr]
    xte = x[ptr:]
    yte = target[ptr:]

    return xtr, ytr, xte, yte


