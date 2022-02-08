import torchvision
from .utils import *
from .gaussian_random_fields import gram_kn, grf_generator
import scipy.io
import math

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
    elif args.pofx == 'sphere':
        x = torch.randn(p, d, device=args.device)
        x /= x.norm(dim=-1, keepdim=True)
    elif args.pofx == "mnist_pca":
        tr = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=True, download=True, transform=transform)
        te = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=False, transform=transform)
        x, target = dataset_to_tensors(list(tr) + list(te))
        assert p <= len(target), "P too large, not enough data-samples!"
        perm = torch.randperm(len(target))[:p]
        x, target = x[perm].to(args.device), target[perm].to(args.device)
        x = center_normalize(x)
        x = pca(x, d, whitening=True)
        target = (2 * (target > 4) - 1)
    elif args.pofx == '1d':
        assert args.d == 2
        mat = scipy.io.loadmat(f'/home/lpetrini/git/regressionsphere/dataset/1d/eric_P{args.ptr}.mat')

        xtr = torch.from_numpy(mat['xd']).to(args.device)[:, 0]
        # ytr = torch.from_numpy(mat['fsd'][:, 0]).to(args.device)
        xte = torch.from_numpy(mat['x'][0]).to(args.device)
        # yte = torch.from_numpy(mat['fs'][:, 0]).to(args.device)

        def angleto2d(xt):
            theta = 2 * math.pi * xt
            return torch.stack([theta.cos(), theta.sin()]).t()

        xtr = angleto2d(xtr)
        xte = angleto2d(xte)
        ytr = torch.ones(len(xtr), device=args.device)
        yte = torch.ones(len(xte), device=args.device)

        return xtr, ytr, xte, yte
    else:
        raise NotImplementedError

    if target is None:
        if args.target == 'norm':
            target = torch.norm(x, dim=1)
        elif 'grf' in args.target:
            teacher_cov = gram_kn(x, x, degree=int(args.target[-1]))
            target = grf_generator(teacher_cov, args.device)
        else:
            raise NotImplementedError

    xtr = x[:ptr]
    ytr = target[:ptr]
    xte = x[ptr:]
    yte = target[ptr:]

    return xtr, ytr, xte, yte


