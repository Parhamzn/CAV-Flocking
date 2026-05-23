"""Alpha-agent action function (port of Phia.m).

phi_alpha(z) = rho_h(z / sigma_norm(r)) * phi(z - sigma_norm(d)).
"""
from flocking_lib.sigma_norm import sigma_norm
from flocking_lib.rho import rho
from flocking_lib.phi import phi


def phi_alpha(z, r, e, d, h, a, b):
    return rho(z / sigma_norm(r, e), h) * phi(z - sigma_norm(d, e), a, b)
