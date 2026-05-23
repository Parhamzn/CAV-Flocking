"""Beta-agent action function (port of PhiBeta.m).

Purely repulsive (Olfati-Saber 2006 eq. 59). Pushes agent away from road-boundary
projection; zero at desired distance d_b.
"""
import numpy as np
from flocking_lib.sigma_norm import sigma_norm
from flocking_lib.rho import rho


def phi_beta(z, d_b, h_b, e):
    d_prime = sigma_norm(d_b, e)
    arg = z - d_prime
    sigma_1 = arg / np.sqrt(1.0 + arg ** 2)
    return rho(z / d_prime, h_b) * (sigma_1 - 1.0)
