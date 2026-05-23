"""Test three τ variants against the compression mode in scenario D
(2x4 vs 2x4 on y_hi=24).

Variants:
  V0. baseline McKenzie velocity-only      u = -c2 * J*(p_i - p_r)
  V1. hybrid: McKenzie + signed radial      u = -c2 * J*(p_i - p_r)
                                              + c3 * (q_i - q_r) / |q_i - q_r|^2
  V2. predictive gating: only apply tau if  closest projected approach < threshold
  V3. flock-relative deflection direction:  magnitude from McKenzie, direction
                                              from agent's position within its own flock
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.flock_layouts import grid_formation
from flocking_lib.control_alpha import control_alpha
from flocking_lib.control_beta import control_beta
from flocking_lib.control_gamma import control_gamma


J = np.array([[0.0, 1.0], [1.0, 0.0]])


def tau_v0_baseline(i, q, p, flock_id, params):
    qi, pi = q[i], p[i]
    ui = np.zeros(2)
    sp = np.linalg.norm(pi)
    if sp < 1e-9: return ui
    dir_i = pi / sp
    for r in range(q.shape[0]):
        if flock_id[r] == flock_id[i]: continue
        diff_q = qi - q[r]
        if np.linalg.norm(diff_q) > params['d_c']: continue
        sr = np.linalg.norm(p[r])
        if sr < 1e-9: continue
        if np.dot(dir_i, p[r] / sr) > 0: continue
        ui -= params['c2_t'] * (J @ (pi - p[r]))
    return ui


def tau_v1_hybrid(i, q, p, flock_id, params):
    qi, pi = q[i], p[i]
    ui = np.zeros(2)
    sp = np.linalg.norm(pi)
    if sp < 1e-9: return ui
    dir_i = pi / sp
    for r in range(q.shape[0]):
        if flock_id[r] == flock_id[i]: continue
        diff_q = qi - q[r]
        nd = np.linalg.norm(diff_q)
        if nd > params['d_c']: continue
        sr = np.linalg.norm(p[r])
        if sr < 1e-9: continue
        if np.dot(dir_i, p[r] / sr) > 0: continue
        ui -= params['c2_t'] * (J @ (pi - p[r]))
        # signed radial repulsion: 1/r^2 falloff in direction of (q_i - q_r)
        d2 = nd ** 2 + 1e-3
        ui += params['c3_t'] * diff_q / d2
    return ui


def tau_v2_predict(i, q, p, flock_id, params):
    qi, pi = q[i], p[i]
    ui = np.zeros(2)
    sp = np.linalg.norm(pi)
    if sp < 1e-9: return ui
    dir_i = pi / sp
    threshold = params['predict_threshold']
    for r in range(q.shape[0]):
        if flock_id[r] == flock_id[i]: continue
        diff_q = qi - q[r]
        if np.linalg.norm(diff_q) > params['d_c']: continue
        diff_p = pi - p[r]
        sr = np.linalg.norm(p[r])
        if sr < 1e-9: continue
        if np.dot(dir_i, p[r] / sr) > 0: continue
        # predict time of closest approach
        denom = np.dot(diff_p, diff_p)
        if denom < 1e-6:
            t_star = 0.0
        else:
            t_star = -np.dot(diff_q, diff_p) / denom
        if t_star < 0:
            t_star = 0.0
        min_vec = diff_q + t_star * diff_p
        if np.linalg.norm(min_vec) > threshold:
            continue
        ui -= params['c2_t'] * (J @ diff_p)
    return ui


def tau_v4_projected_radial(i, q, p, flock_id, params):
    """McKenzie velocity term + radial repulsion gated to only contribute in y
    when it agrees with the McKenzie v-term sign — never opposes it."""
    qi, pi = q[i], p[i]
    ui = np.zeros(2)
    sp = np.linalg.norm(pi)
    if sp < 1e-9: return ui
    dir_i = pi / sp
    for r in range(q.shape[0]):
        if flock_id[r] == flock_id[i]: continue
        diff_q = qi - q[r]
        nd = np.linalg.norm(diff_q)
        if nd > params['d_c']: continue
        sr = np.linalg.norm(p[r])
        if sr < 1e-9: continue
        if np.dot(dir_i, p[r] / sr) > 0: continue
        v_term = -params['c2_t'] * (J @ (pi - p[r]))
        ui += v_term
        d2 = nd ** 2 + 1e-3
        radial_y = params['c3_t'] * diff_q[1] / d2
        # Only add radial_y if it has the SAME sign as v_term[1].
        if v_term[1] != 0 and np.sign(radial_y) == np.sign(v_term[1]):
            ui[1] += radial_y
    return ui


def tau_v3_flock_relative(i, q, p, flock_id, params):
    qi, pi = q[i], p[i]
    ui = np.zeros(2)
    sp = np.linalg.norm(pi)
    if sp < 1e-9: return ui
    dir_i = pi / sp
    # agent's y-offset within its own flock
    own_mask = (flock_id == flock_id[i])
    own_y_center = q[own_mask, 1].mean()
    y_offset = qi[1] - own_y_center
    # deflection sign: +1 if above own centroid (go up), -1 if below
    if abs(y_offset) < 0.1:
        deflect_sign = 0.0  # near-centroid -> fall back to McKenzie
    else:
        deflect_sign = np.sign(y_offset)
    for r in range(q.shape[0]):
        if flock_id[r] == flock_id[i]: continue
        diff_q = qi - q[r]
        if np.linalg.norm(diff_q) > params['d_c']: continue
        diff_p = pi - p[r]
        sr = np.linalg.norm(p[r])
        if sr < 1e-9: continue
        if np.dot(dir_i, p[r] / sr) > 0: continue
        if deflect_sign == 0:
            ui -= params['c2_t'] * (J @ diff_p)
        else:
            mag = params['c2_t'] * abs(diff_p[0])   # magnitude from x-component
            ui[1] += mag * deflect_sign
    return ui


def run_with_tau(tau_fn, flock_inits, params, y_lo, y_hi, T, TS, a_max=9.0):
    num_steps = int(round(T / TS)) + 1
    N_total = sum(q.shape[0] for q, _ in flock_inits)
    q = np.zeros((N_total, 2, num_steps))
    p = np.zeros((N_total, 2, num_steps))
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
                     + tau_fn(i, qt, pt, flock_id, params))
        for i in range(N_total):
            mag = np.linalg.norm(ut[i])
            if mag > a_max:
                ut[i] = a_max * ut[i] / mag
        p[:, :, t + 1] = pt + TS * ut
        q[:, :, t + 1] = qt + TS * p[:, :, t + 1]
    return q, p, flock_id


def metrics(q, flock_id, y_hi, d_b):
    N = q.shape[0]
    inter_min = np.inf; intra_min = np.inf
    for t in range(q.shape[2]):
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(q[i, :, t] - q[j, :, t])
                if flock_id[i] != flock_id[j]:
                    if d < inter_min: inter_min = d
                else:
                    if d < intra_min: intra_min = d
    mask1, mask2 = (flock_id == 1), (flock_id == 2)
    x1 = q[mask1, 0, :].mean(axis=0); x2 = q[mask2, 0, :].mean(axis=0)
    cross_idx = int(np.argmin(np.abs(x1 - x2)))
    y_gap = abs(q[mask2, 1, cross_idx].mean() - q[mask1, 1, cross_idx].mean())
    # row sep within flock 1 (lower 4 are row 0, upper 4 are row 1)
    y_r0 = q[:4, 1, :].mean(axis=0); y_r1 = q[4:8, 1, :].mean(axis=0)
    row_sep_min = (y_r1 - y_r0).min()
    return dict(inter=inter_min, intra=intra_min, clearance=y_gap,
                row_sep_min=row_sep_min)


def main():
    params = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,'d_c':40.0,'c1_t':0.0,'c2_t':0.08,
        'c3_t': 80.0,                # radial repulsion gain (V1)
        'predict_threshold': 4.0,    # m, for V2
        'p_d_flock1': np.array([+10.0, 0.0]),
        'p_d_flock2': np.array([-10.0, 0.0]),
    }
    s = params['d_a']
    v1, v2 = params['p_d_flock1'], params['p_d_flock2']
    q1, p1 = grid_formation(2, 4, x_center=-20 - 1.5*s, y_center=12.0, spacing=s, vel=v1)
    q2, p2 = grid_formation(2, 4, x_center=+20 + 1.5*s, y_center=12.0, spacing=s, vel=v2)

    variants = [
        ('V0 · McKenzie baseline',          tau_v0_baseline),
        ('V1 · hybrid (vel + radial)',      tau_v1_hybrid),
        ('V4 · vel + projected radial',     tau_v4_projected_radial),
        ('V4 · vel + projected radial (c3↓)', tau_v4_projected_radial),
    ]

    results = []
    for label, tau_fn in variants:
        # Use a smaller c3_t for the second V4 entry
        local_params = dict(params)
        if '(c3↓)' in label:
            local_params['c3_t'] = 20.0
        q, p, fid = run_with_tau(tau_fn, [(q1, p1), (q2, p2)], local_params,
                                 y_lo=0.0, y_hi=24.0, T=12.0, TS=0.02)
        m = metrics(q, fid, 24.0, local_params['d_b'])
        results.append((label, q, p, fid, m))
        print(f'{label:42s}  inter={m["inter"]:5.2f}  intra={m["intra"]:5.2f}  '
              f'row_sep_min={m["row_sep_min"]:5.2f}  clearance={m["clearance"]:5.2f}')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for k, (label, q, p, fid, m) in enumerate(results):
        ax = axes[k // 2, k % 2]
        N = q.shape[0]
        for i in range(N):
            xs, ys = q[i, 0, :], q[i, 1, :]
            col = 'b' if fid[i] == 1 else 'r'
            ax.plot(xs, ys, color=col, linewidth=1, alpha=0.7)
            ax.plot(xs[0], ys[0], marker='o', color=col, markersize=6)
            ax.plot(xs[-1], ys[-1], marker='s', color=col, markersize=6, fillstyle='none')
        ax.axhline(0, color='k', linewidth=1.5); ax.axhline(24, color='k', linewidth=1.5)
        ax.set_xlim(q[:, 0, :].min() - 5, q[:, 0, :].max() + 5)
        ax.set_ylim(-1, 25); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title(f'{label}\ninter={m["inter"]:.2f}  intra={m["intra"]:.2f}  '
                     f'row_sep_min={m["row_sep_min"]:.2f}',
                     fontsize=10)
    fig.suptitle('τ variants tested on scenario D (2x4 vs 2x4, y_hi=24)', y=0.995)
    fig.tight_layout()
    fig.savefig('test_tau_variants.png', dpi=110, bbox_inches='tight')
    print('saved test_tau_variants.png')


if __name__ == '__main__':
    main()
