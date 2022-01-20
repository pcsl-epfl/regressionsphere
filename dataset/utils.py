import torch


def pca(x, d, whitening):
    """
    :param x: [P, ...]
    :return: [P, d]
    """

    z = x.flatten(1)
    mu = z.mean(0)
    cov = (z - mu).t() @ (z - mu) / len(z)

    val, vec = torch.linalg.eigh(cov)
    val, idx = val.sort(descending=True)
    vec = vec[:, idx]

    u = (z - mu) @ vec[:, :d]
    if whitening:
        u.mul_(val[:d].rsqrt())
    else:
        u.mul_(val[:d].mean().rsqrt())

    return u

def dataset_to_tensors(dataset):
    dataset = [(x.type(torch.float64), int(y)) for x, y in dataset]
    x = torch.stack([x for x, y in dataset])
    y = torch.tensor([y for x, y in dataset], dtype=torch.long)
    return x, y


def center_normalize(x):
    x = x - x.mean(0)
    x = (x[0].numel() ** 0.5) * x / x.flatten(1).norm(dim=1).view(-1, *(1,) * (x.dim() - 1))
    return x
