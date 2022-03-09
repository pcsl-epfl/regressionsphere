import torch
import math


# Kernel functions `kn` for ReLU^n net
def k0(phi):
    return (math.pi - phi) / (math.pi)


def k1(phi):
    return (torch.sin(phi) + (math.pi - phi) * torch.cos(phi)) / (math.pi)


def k2(phi):
    return (
        3.0 * torch.sin(phi) * torch.cos(phi)
        + (math.pi - phi) * (1.0 + 2.0 * torch.cos(phi) * torch.cos(phi))
    ) / math.pi


def k3(phi):
    return (
        3.0 * torch.sin(phi) * torch.sin(phi) * torch.sin(phi)
        - 4.0 * torch.cos(phi) * torch.cos(phi) * torch.sin(phi)
        + (math.pi - phi)
        * (
            2.0 * torch.cos(phi) * torch.sin(phi) * torch.sin(phi)
            - torch.cos(phi)
            - 2.0 * torch.cos(phi) * torch.cos(phi) * torch.cos(phi)
        )
    ) / math.pi


def gaussian(phi):
    return (-(1 - phi)).exp()


def laplace(phi, c=0.001):
    return torch.exp(-c * torch.sqrt(1.0 - phi))


def gram_kn(X, Y, degree):
    """
    Compute the one-hidden-layer ReLU^deg.-net random-features gram matrix.
    :param X: torch.tensor of shape (N, d)
    :param Y: torch.tensor of shape (N, d)
    :param degree: return RF kernel given by ReLU^degree net.
    :return: gram matrix (N, N)
    """
    assert Y.size(1) == X.size(1), "data have different dimension!"

    kernels = {0: k0, 1: k1, 2: k2, 3: k3, -2: gaussian, -1: laplace}
    kn = kernels[degree]

    XdY = X @ Y.t()
    phi = torch.arccos(torch.clamp(XdY, min=-1.0, max=1.0)) if degree > -1 else XdY
    gram = kn(phi)

    return gram


def gram_ntk(X, Y):
    """
    X, Y tensors of shape (P, d)
    """
    assert Y.size(1) == X.size(1), "data have different dimension!"

    XdY = X @ Y.t()
    XnYn = X.norm(dim=1, keepdim=True) @ Y.norm(dim=1, keepdim=True).t()
    phi = torch.arccos(torch.clamp(XdY / XnYn, min=-1.0, max=1.0))
    gram = (XdY * k0(phi) + XnYn * k1(phi)) / (2.0 * X.size(1))

    return gram


def grf_generator(gram):
    """
    Sample a Gaussian random field.
    :param gram: field covariance matrix
    :return: Gaussian random field with given covariance.
    """
    assert gram.shape[-1] == gram.shape[-2], "Covariance matri must be squared!"
    N = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(len(gram), device=gram.device), gram
    )
    y = N.sample()
    return y


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
    alpha = (
        torch.linalg.inv(K_trtr + ridge * torch.eye(y_tr.size(0), device=K_trtr.device))
        @ y_tr
    )
    f = K_tetr @ alpha
    mse = (f - y_te).pow(2).mean()
    return mse
