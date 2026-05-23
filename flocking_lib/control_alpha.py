"""f_alpha for agent i (port of ControlAlpha.m).

Gradient (lattice) term + velocity consensus over same-flock alpha-neighbors.
Olfati-Saber 2006 eq. 23.
"""
import numpy as np
from flocking_lib.sigma_norm import sigma_norm
from flocking_lib.n_ij import n_ij
from flocking_lib.a_ij import a_ij
from flocking_lib.phi_alpha import phi_alpha


def control_alpha(i, q, p, flock_id, params):
    N = q.shape[0]
    qi = q[i]
    pi = p[i]
    grad = np.zeros(2)
    cons = np.zeros(2)
    r_sigma = sigma_norm(params['r_a'], params['e'])
    for j in range(N):
        if j == i:
            continue
        if flock_id[j] != flock_id[i]:
            continue
        qj = q[j]
        pj = p[j]
        z = sigma_norm(qj - qi, params['e'])
        if z > r_sigma:
            continue
        grad = grad + phi_alpha(z, params['r_a'], params['e'], params['d_a'],
                                params['h_a'], params['a'], params['b']) \
                      * n_ij(qi, qj, params['e'])
        cons = cons + a_ij(qi, qj, params['h_a'], params['r_a'], params['e']) * (pj - pi)
    return params['c1_a'] * grad + params['c2_a'] * cons
