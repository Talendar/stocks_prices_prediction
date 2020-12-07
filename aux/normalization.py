""" Auxiliary functions for normalizing data.

@author Gabriel Nogueira (Talendar)
"""


def min_max_norm(data, dmin=None, dmax=None):
    if dmin is None:
        dmin = data.min(axis=0)

    if dmax is None:
        dmax = data.max(axis=0)

    return (data - dmin) / (dmax - dmin), \
           dmin, dmax


def min_max_denorm(data, dmin, dmax):
    return (data * (dmax - dmin)) + dmin


def zscore_norm(data, mean=None, std=None):
    if mean is None:
        mean = data.mean(axis=0)

    if std is None:
        std = data.std(axis=0)

    return (data - mean) / std, \
           mean, std


def zscore_denorm(data, mean, std):
    return data * std + mean
