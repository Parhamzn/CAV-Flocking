"""Neighbour check (port of Neighbour.m).

Returns 1 if |qref - qquery| <= r, else 0.
"""
import numpy as np


def neighbour(qref, qquery, r):
    qref = np.asarray(qref, dtype=float)
    qquery = np.asarray(qquery, dtype=float)
    return 1 if np.linalg.norm(qref - qquery) <= r else 0
