"""Tau-agent action function (port of PhiTau.m).

Purely repulsive interaction with members of the OPPOSING flock. Project's
novel agent type per the Week-2 deck.
"""
import numpy as np
from flocking_lib.sigma_norm import sigma_norm
from flocking_lib.rho import rho


def phi_tau(z, d_t, h_t, e):
    d_prime = sigma_norm(d_t, e)
    arg = z - d_prime
    sigma_1 = arg / np.sqrt(1.0 + arg ** 2)
    return rho(z / d_prime, h_t) * (sigma_1 - 1.0)
