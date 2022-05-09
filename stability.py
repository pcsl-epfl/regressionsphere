import torch
import sys
sys.path.insert(0, '/home/lpetrini/git/diffeomorphism/')
from diff import deform
import math


### Images ###

def deformation_and_noise_stability(f, imgs):
    """
    Measure deformation and noise stability of a fully-connected network `f` on `imgs`.
    :param f: FCN predictor function
    :param imgs: batch of image samples
    :return: deformation and noise stability averaged over the batch and perturbations.
    """
    ds = diffeos(imgs)
    ns = perturb(imgs, ds)
    imgs, ns, ds = imgs.flatten(1), ns.flatten(1), ds.flatten(1)
    D = stability(f, imgs, [ds])
    G = stability(f, imgs, [ns])
    return D[0], G[0]

def diffeos(imgs, delta=1, c=3, interp='linear'):
    """
    Deform a batch of images.
    :param imgs: batch of images
    :param delta: average pixel displacement
    :param c: high-frequency cut-off
    :param interp: interpolation method
    :return: batch of *deformed* images
    """
    n = imgs.shape[-1]
    T = typical_temperature(delta, c, n)
    return torch.stack([deform(i, T, c, interp) for i in imgs])

def perturb(imgs, timgs):
    """
    :param imgs: original images
    :param timgs: locally translated images (~diffeo)
    :return: original images, locally translated images, noisy images
    """
    sigma = (timgs - imgs).pow(2).sum([1, 2, 3], keepdim=True).sqrt()
    eta = torch.randn(imgs.shape, device=imgs.device)
    eta = eta / eta.pow(2).sum([1, 2, 3], keepdim=True).sqrt() * sigma
    nimgs = imgs + eta
    return nimgs


### Spherical data ###

def rotation_stability(f, x, angle):
    """
    Measure the stability and of a fully-connected network `f` to random rotations of `x` by `angle`.
    :param f: predictor
    :param x: data batch
    :param angle: angle in degrees
    :return: rotation stability averaged over the batch of samples and angles.
    """
    rx = random_direction_rotation(x, angle)
    return stability(f, x, [rx])

def random_direction_rotation(x, angle):
    """
    Rotate the vector `x` toward a random `direction` by `angle` degrees.
    """
    direction = torch.randn(x.shape, device=x.device)
    return high_d_rotation(x, direction, angle)

def high_d_rotation(x, direction, angle):
    """
    Rotate the high-d vector `x` toward `direction` by `angle` degrees.
    """
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle, dtype=float)
    angle *= 2 * math.pi / 360
    xp = x.div(x.norm(dim=-1, keepdim=True))
    B = direction - (xp * direction).sum(dim=-1, keepdim=True) * xp
    B /= B.norm(dim=-1, keepdim=True)
    rx = (angle.cos() * xp + angle.sin() * B)
    return rx.mul(x.norm(dim=-1, keepdim=True))


### Utils ###

def stability(f, i, ns):
    """
    compute stability of the function `f` to perturbations `ns` of `i`
    :param f: network function
    :param i: original image(s)
    :param ns: tensor of perturbed batches of images
    :return: stabilities
    """
    with torch.no_grad():
        f0 = f(i).detach().reshape(len(i), -1)  # [batch, ...]
        deno = torch.cdist(f0, f0).pow(2).mean().item() + 1e-10
        S = []
        for n in ns:
            fn = f(n).detach().reshape(len(i), -1)  # [batch, ...]
            S += [
                (fn - f0).pow(2).mean(0).sum().item() / deno
            ]
        return torch.tensor(S)

def typical_temperature(delta, cut, n):
    if isinstance(cut, (float, int)):
        log = math.log(cut)
    else:
        log = cut.log()
    return 4 * delta ** 2 / (math.pi * n ** 2 * log)