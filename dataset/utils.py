"""Utility functions adapted from mariogeiger/feature_lazy."""

import torch
from itertools import chain


def pca(x, d, whitening):
    """
    :param x: [P, ...]
    :return: [P, d]
    """

    z = x.flatten(1)
    mu = z.mean(0)
    cov = (z - mu).t() @ (z - mu) / len(z)

    val, vec = cov.symeig(eigenvectors=True)
    val, idx = val.sort(descending=True)
    vec = vec[:, idx]

    u = (z - mu) @ vec[:, :d]
    if whitening:
        u.mul_(val[:d].rsqrt())
    else:
        u.mul_(val[:d].mean().rsqrt())

    return u

def dataset_to_tensors(dataset):
    dataset = [(x.type(torch.float64), int(y), i) for i, (x, y) in enumerate(dataset)]
    x = torch.stack([x for x, y, i in dataset])
    y = torch.tensor([y for x, y, i in dataset], dtype=torch.long)
    i = torch.tensor([i for x, y, i in dataset], dtype=torch.long)
    return x, y, i


def intertwine_labels(x, y, i):
    classes = y.unique()
    sets = [(x[y == z], y[y == z], i[y == z]) for z in classes]

    del x, y, i

    sets = [
        (x[rp], y[rp], i[rp])
        for x, y, i, rp in
        ((x, y, i, torch.randperm(len(x))) for x, y, i in sets)
    ]

    x = torch.stack(list(chain(*zip(*(x for x, y, i in sets)))))
    y = torch.stack(list(chain(*zip(*(y for x, y, i in sets)))))
    i = torch.stack(list(chain(*zip(*(i for x, y, i in sets)))))
    return x, y, i


def center_normalize(x):
    x = x - x.mean(0)
    x = (x[0].numel() ** 0.5) * x / x.flatten(1).norm(dim=1).view(-1, *(1,) * (x.dim() - 1))
    return x


def intertwine_split(x, y, i, ps, seeds, classes):
    assert len(ps) == len(seeds)

    if len(ps) == 0:
        return []

    xs = [(x[y == z], i[y == z]) for z in classes]

    ps = list(ps)
    seeds = list(seeds)

    p = ps.pop(0)
    seed = seeds.pop(0)

    torch.manual_seed(seed)
    xx = []
    ii = []
    for x, i in xs:
        rp = torch.randperm(len(x))
        xx.append(x[rp])
        ii.append(i[rp])

    ys = [torch.full((len(x),), z, dtype=torch.long) for x, z in zip(xx, classes)]

    x = torch.stack(list(chain(*zip(*xx))))
    y = torch.stack(list(chain(*zip(*ys))))
    i = torch.stack(list(chain(*zip(*ii))))

    assert len(x) >= p, "only {} elements in this dataset, asking for {}".format(len(x), p)
    return [(x[:p], y[:p], i[:p])] + intertwine_split(x[p:], y[p:], i[p:], ps, seeds, classes)