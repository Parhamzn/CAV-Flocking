"""Bump function rho_h(z) (port of rho.m).

1 on [0, h), cosine taper on [h, 1], 0 elsewhere.
"""
import numpy as np


def rho(z, h):
    if 0.0 <= z < h:
        return 1.0
    elif h <= z <= 1.0:
        return 0.5 * (1.0 + np.cos(np.pi * (z - h) / (1.0 - h)))
    else:
        return 0.0
