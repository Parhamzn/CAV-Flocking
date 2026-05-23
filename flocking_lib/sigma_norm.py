"""Sigma-norm smoothing function (port of SigmaNorm.m).

sigma_eps(z) = (1/eps) * (sqrt(1 + eps * |z|^2) - 1).
"""
import numpy as np


def sigma_norm(z, e):
    z = np.asarray(z, dtype=float)
    return (1.0 / e) * (np.sqrt(1.0 + e * np.dot(z, z)) - 1.0)
