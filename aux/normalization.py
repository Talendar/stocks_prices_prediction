""" Auxiliary functions for normalizing data.

@author Gabriel Nogueira (Talendar)
"""


def min_max_norm(data, dmin=None, dmax=None):
    if dmin is None:
        dmin = data.min(axis=0)

    if dmax is None:
        dmax = data.max(axis=0)

    return (data - dmin) / (dmax - dmin), dmin, dmax


def min_max_denorm(data, dmin, dmax):
    return (data * (dmax - dmin)) + dmin
