"""f_tau for agent i — coordinative interaction with opposing flocks.

Per McKenzie (2012), eq. 6.4:

    u_tau_i = - c_tau1 * sigma_2(q_i - q_r) * J * (q_i - q_r)
              - c_tau2 * sigma_2(p_i - p_r) * J * (p_i - p_r)

with the swap matrix J = [[0, 1], [1, 0]] (reflection across y = x), and the
step gate

    sigma_2 = 1   iff   |q_i - q_r| <= d_c
                        and the agents' headings differ by >= 90 deg
                        (i.e. dot(p_hat_i, p_hat_r) <= 0).

For head-on encounters, J * (q_i - q_r) is purely lateral, so the tau-force
pushes agents *across* the road rather than back at each other — that's what
splits two opposing flocks into different lanes.

The summation is over every agent r in any opposing flock that satisfies the
gate. The paper writes a single r in the equation; we read this as an implicit
sum, matching the alpha/beta convention.

NOTE — empirical finding: McKenzie's position term `c_tau1 * J * (q_i - q_r)`
flips sign at closest approach, and in the first half of the encounter it
actively opposes the velocity term's intended deflection direction. At low
v_d the position term overpowers the velocity term and the algorithm fails
to deflect at all (inter-flock min drops to 0.95 m at v_d=3). Setting
c_tau1 = 0 (velocity-only variant) removes the failure mode and improves the
inter-flock minimum monotonically across the entire speed range. The driver
scripts default to c_tau1 = 0 for this reason. The full formula is preserved
in the implementation so the original paper's behavior is reproducible.
"""
import numpy as np


J = np.array([[0.0, 1.0],
              [1.0, 0.0]])

# True 90° rotation matrix (CCW). Substituting this for J generalizes
# McKenzie's lateral deflection to flocks moving in any direction. For
# axis-aligned 1D opposing flocks (±x) J and R produce identical force,
# so this is backward-compatible on the canonical scenario.
R_PLUS_90  = np.array([[0.0, -1.0],
                       [1.0,  0.0]])
R_MINUS_90 = np.array([[0.0,  1.0],
                       [-1.0, 0.0]])


def control_tau(i, q, p, flock_id, params):
    qi = q[i]
    pi_v = p[i]
    ui = np.zeros(2)

    speed_i = np.linalg.norm(pi_v)
    if speed_i < 1e-9:
        return ui                                # heading undefined
    dir_i = pi_v / speed_i

    d_c = params['d_c']
    # Optional predictive suppression threshold: skip the pair if straight-line
    # projection has them passing further than this distance apart at closest
    # approach. None disables it (default).
    suppress_th = params.get('predict_suppress_threshold')
    # Optional override of McKenzie's reflection matrix J with a true rotation
    # (R_PLUS_90 or R_MINUS_90). For axis-aligned 1D flocks J and R produce
    # the same force; for non-aligned multi-flock geometries they differ.
    M = params.get('tau_matrix', J)

    N = q.shape[0]
    for r in range(N):
        if flock_id[r] == flock_id[i]:
            continue
        diff_q = qi - q[r]
        if np.linalg.norm(diff_q) > d_c:
            continue
        speed_r = np.linalg.norm(p[r])
        if speed_r < 1e-9:
            continue
        if np.dot(dir_i, p[r] / speed_r) > 0:    # headings parallel-ish — skip
            continue
        diff_p = pi_v - p[r]
        if suppress_th is not None:
            denom = float(np.dot(diff_p, diff_p))
            if denom > 1e-9:
                t_star = -float(np.dot(diff_q, diff_p)) / denom
                if t_star < 0:
                    t_star = 0.0
            else:
                t_star = 0.0
            min_vec = diff_q + t_star * diff_p
            if np.linalg.norm(min_vec) > suppress_th:
                continue                          # projected pass is clear — skip
        ui = ui - params['c1_t'] * (M @ diff_q) - params['c2_t'] * (M @ diff_p)
    return ui
