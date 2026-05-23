"""Generic multi-flock simulator with arbitrary initial conditions.

Accepts a list of (q_init, p_init) per flock and runs the alpha + beta +
gamma + tau control law on the combined system. Returns full state history.
"""
import numpy as np

from flocking_lib.control_alpha import control_alpha
from flocking_lib.control_beta import control_beta
from flocking_lib.control_gamma import control_gamma
from flocking_lib.control_tau import control_tau


def run_multi_flock(flock_inits, params, y_lo, y_hi, T, TS, a_max=9.0):
    """Run the alpha+beta+gamma+tau sim with arbitrary flock geometries.

    Parameters
    ----------
    flock_inits : list of (q_init, p_init)
        One entry per flock. Each q_init is (N_k, 2), p_init is (N_k, 2).
    params : dict
        Algorithm parameters. Must include 'p_d_flock1', 'p_d_flock2'.
        Currently supports up to two flocks (gamma feeds the per-flock desired
        velocity by flock id).

    Returns
    -------
    q, p, u : (N_total, 2, num_steps) arrays
    flock_id : (N_total,) array of 1-based flock indices
    """
    num_steps = int(round(T / TS)) + 1
    N_total = sum(q.shape[0] for q, _ in flock_inits)

    q = np.zeros((N_total, 2, num_steps))
    p = np.zeros((N_total, 2, num_steps))
    u = np.zeros((N_total, 2, num_steps))
    flock_id = np.zeros(N_total, dtype=int)

    offset = 0
    for k, (qi, pi) in enumerate(flock_inits, start=1):
        n = qi.shape[0]
        q[offset:offset + n, :, 0] = qi
        p[offset:offset + n, :, 0] = pi
        flock_id[offset:offset + n] = k
        offset += n

    for t in range(num_steps - 1):
        qt, pt = q[:, :, t], p[:, :, t]
        ut = np.zeros((N_total, 2))
        for i in range(N_total):
            ut[i] = (control_alpha(i, qt, pt, flock_id, params)
                     + control_beta (i, qt, pt, y_lo, y_hi, params)
                     + control_gamma(i, qt, pt, flock_id, params)
                     + control_tau  (i, qt, pt, flock_id, params))
        for i in range(N_total):
            mag = np.linalg.norm(ut[i])
            if mag > a_max:
                ut[i] = a_max * ut[i] / mag
        u[:, :, t] = ut
        p[:, :, t + 1] = pt + TS * ut
        q[:, :, t + 1] = qt + TS * p[:, :, t + 1]

    return q, p, u, flock_id


def encounter_metrics(q, flock_id, y_hi, d_b):
    """inter-flock min, intra-flock min, wall-proximity fraction, clearance@x0."""
    N = q.shape[0]
    inter_min = np.inf
    intra_min = np.inf
    for t in range(q.shape[2]):
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(q[i, :, t] - q[j, :, t])
                if flock_id[i] != flock_id[j]:
                    if d < inter_min:
                        inter_min = d
                else:
                    if d < intra_min:
                        intra_min = d
    near_wall = ((q[:, 1, :] < d_b) | (q[:, 1, :] > y_hi - d_b))
    wall_proximity_fraction = near_wall.mean()
    # clearance at moment of x-encounter (centroids share an x)
    mask1 = flock_id == 1; mask2 = flock_id == 2
    x1 = q[mask1, 0, :].mean(axis=0)
    x2 = q[mask2, 0, :].mean(axis=0)
    cross_idx = int(np.argmin(np.abs(x1 - x2)))
    y1 = q[mask1, 1, cross_idx].mean()
    y2 = q[mask2, 1, cross_idx].mean()
    clearance = abs(y2 - y1)
    return dict(inter_min=inter_min, intra_min=intra_min,
                wall_proximity_fraction=wall_proximity_fraction,
                clearance_at_x0=clearance)
