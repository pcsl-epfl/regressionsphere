import torchvision
from .utils import *
from .gaussian_random_fields import gram_kn, grf_generator
from .teacher import FCTeacher
import sys

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
        x = torch.randn(p, d, device=args.device) / d ** .5
    elif args.pofx == 'uniform':
        u = torch.randn(p, d, device=args.device)
        norm = u.norm(dim=1, keepdim=True)
        r = torch.rand(p, 1).pow(1 / d)
        x = r * u / norm
    elif args.pofx == 'sphere':
        try:
            if 'pyth-clone' in sys.path[1]:
                print('loading data from izar...')
                trainset_path = f'/work/pcsl-fdf/spherical-model/trainsets/trainset_randunif_d{d}_n{ptr}_{args.dataseed}.pt'
            else:
                print('loading data from pcslsrv...')
                trainset_path = f'/home/lpetrini/data/spherical-model/trainset_randunif_d{d}_n{ptr}_{args.dataseed}.pt'
            x = torch.load(trainset_path).to(args.device)
            x = torch.cat([x, torch.randn(pte, d, device=args.device)])
        except FileNotFoundError:
            print('Input data file not found!!')
            x = torch.randn(p, d, device=args.device)
            x /= x.norm(dim=-1, keepdim=True)
    elif args.pofx == 'spread':
        trainset_path = f'/home/lpetrini/data/spherical-model/trainset_randspread_div{args.div:.01f}_d{d}_n{ptr}_{args.dataseed}.pt'
        x = torch.load(trainset_path).to(args.device)
        x = torch.cat([x, torch.randn(pte, d, device=args.device)])
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
    elif args.pofx == "mnist":
        tr = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=True, download=True,
                                               transform=transform)
        te = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=False, transform=transform)
        x, target = dataset_to_tensors(list(tr) + list(te))
        assert p <= len(target), "P too large, not enough data-samples!"
        perm = torch.randperm(len(target))[:p]
        x, target = x[perm], target[perm]
        x = center_normalize(x).flatten(-2).to(args.device)
        target = torch.tensor([t in [0, 1, 2, 3, 4] for t in target], dtype=int, device=args.device).mul(2).add(-1)
    elif args.pofx == "fashion-mnist":
        tr = torchvision.datasets.FashionMNIST('~/.torchvision/datasets/FashionMNIST', train=True, download=True, transform=transform)
        te = torchvision.datasets.FashionMNIST('~/.torchvision/datasets/FashionMNIST', train=False, transform=transform)
        x, target = dataset_to_tensors(list(tr) + list(te))
        assert p <= len(target), "P too large, not enough data-samples!"
        perm = torch.randperm(len(target))[:p]
        x, target = x[perm], target[perm]
        x = center_normalize(x).flatten(-2).to(args.device)
        target = torch.tensor([t in [0, 2, 3, 4, 6] for t in target], dtype=int, device=args.device).mul(2).add(-1)
    else:
        raise NotImplementedError

    if target is None:
        if args.target == 'norm':
            target = torch.norm(x, dim=1)
        elif 'grf' in args.target:
            teacher_cov = gram_kn(x, x, degree=int(args.target[-1]))
            target = grf_generator(teacher_cov, args.device)
        elif args.target == 'teacher':
            torch.manual_seed(0)
            t = FCTeacher(d=args.d, h=1e7, a=args.act_power, act=args.teacher_act, device=args.device)
            target = t(x)
        else:
            raise NotImplementedError

    xtr = x[:ptr]
    ytr = target[:ptr]
    xte = x[ptr:]
    yte = target[ptr:]

    return xtr, ytr, xte, yte


