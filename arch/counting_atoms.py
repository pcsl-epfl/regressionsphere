"""
    Assumed that the features measure becomes atomic, compute the number of atoms in one-hidden-layer neural networks.
"""

import torch
from collections import defaultdict

def compute_atoms_norm(xtr, w1, w2):
    """
        :param xtr: training set
        :param w1:  1st layer weights
        :param w2:  2nd layer weights
        :return: torch.tensor w/ `len(a_norm) = n. attr.` and `a_norm[i]` = norm of attractor `i`.
    """
    hashed_activations = activations2hash(xtr, w1, w2)
    a_dict = atoms_dict(hashed_activations)
    a_norm, _ = atoms_norm(a_dict, w1.norm(dim=-1) * w2)
    return a_norm

def activations2hash(xtr, w1, w2):
    return [hash(wi.numpy().tobytes()) for wi in (w1[w2[0] != 0] @ xtr.t()).sign()]

def atoms_dict(hashed_activations):
    return dict(sorted(list_duplicates(hashed_activations)))

def atoms_norm(a_dict, w2):
    a_norm = dict()
    for el in a_dict:
        a_norm[el] = w2[0, a_dict[el]].sum().abs()
    return torch.stack([v for v in a_norm.values()]), list(a_norm.keys())

def list_duplicates(seq):
    count = defaultdict(list)
    for i, item in enumerate(seq):
        count[item].append(i)
    return ((key, locs) for key, locs in count.items())
