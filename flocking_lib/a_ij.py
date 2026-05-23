"""Spatial adjacency matrix element (port of aij.m).

a_ij = rho_h(sigma_norm(qj - qi) / sigma_norm(r)).
"""
import numpy as np
from flocking_lib.sigma_norm import sigma_norm
from flocking_lib.rho import rho


def a_ij(qi, qj, h, r, e):
    qi = np.asarray(qi, dtype=float)
    qj = np.asarray(qj, dtype=float)
    return rho(sigma_norm(qj - qi, e) / sigma_norm(r, e), h)
