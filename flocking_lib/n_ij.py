"""Direction vector n_ij (port of nij.m).

n_ij = (qj - qi) / sqrt(1 + eps * |qj - qi|^2).
"""
import numpy as np


def n_ij(qi, qj, e):
    qi = np.asarray(qi, dtype=float)
    qj = np.asarray(qj, dtype=float)
    diff = qj - qi
    return diff / np.sqrt(1.0 + e * np.dot(diff, diff))
