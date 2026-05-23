"""f_gamma for agent i (port of ControlGamma.m).

Navigational feedback to the flock's desired velocity. Simplified Olfati-Saber
gamma: velocity feedback (rather than a moving virtual leader q_gamma(t))
plus an *optional* position-feedback term toward a target y-band.

Velocity feedback (always on):
    u_vel = -c_g * (p_i - p_d[flock])

Position feedback toward target y-band (on iff params has both 'c_g_pos'
and 'y_target_per_flock'):
    u_pos.y = -c_g_pos * (q_i.y - y_target[flock])

The position term is intentionally restricted to y only — it represents an
externally-assigned target *lane* (band of y), not a target point. x is left
to the velocity term.

Supports an arbitrary number of flocks via params['p_d_per_flock'][flock_id]
and params['y_target_per_flock'][flock_id]. The 2-flock keys remain for
backward compatibility.
"""
import numpy as np


def control_gamma(i, q, p, flock_id, params):
    pi = p[i]
    qi = q[i]
    # ---- velocity feedback (always on) ------------------------------------
    p_d_dict = params.get('p_d_per_flock')
    if p_d_dict is not None:
        p_d = p_d_dict[flock_id[i]]
    else:
        p_d = params['p_d_flock1'] if flock_id[i] == 1 else params['p_d_flock2']
    u = -params['c_g'] * (pi - np.asarray(p_d))

    # ---- optional position feedback toward target y-band ------------------
    y_target_dict = params.get('y_target_per_flock')
    c_g_pos = params.get('c_g_pos', 0.0)
    if y_target_dict is not None and c_g_pos > 0:
        y_target = y_target_dict[flock_id[i]]
        u[1] += -c_g_pos * (qi[1] - y_target)
    return u
