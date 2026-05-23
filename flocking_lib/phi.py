"""Uneven sigmoidal phi(z) (port of Phi.m; Olfati-Saber 2006 eq. 15).

phi(z) = 0.5 * [(a+b) * sigma_1(z+c) + (a-b)],  c = |a-b|/sqrt(4ab),  0 < a <= b.
"""
import numpy as np


def phi(z, a, b):
    c = abs(a - b) / np.sqrt(4.0 * a * b)
    sigma_1 = (z + c) / np.sqrt(1.0 + (z + c) ** 2)
    return 0.5 * ((a + b) * sigma_1 + (a - b))
