"""Stagger sweeps for the intersection and merge scenarios.

  · Intersection: keep E and W flocks at standard distance from the
    intersection, push N and S flocks back by `stagger` m so they arrive
    at the box later. Sweep stagger.
  · Merge: vary the on-ramp's initial offset relative to the main flock.

For each sweep we measure inter-flock min, off-road incidents, and report
the threshold where the system transitions from "unsafe" (below car width)
to "safe."
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.control_tau import R_PLUS_90
from exp_cross_intersection import make_cross_flock, draw_road_geometry
from exp_cross_asymmetric import run_cross_asym, metrics as cross_metrics
from exp_merge_asymmetric import run_asym_merge, metrics as merge_metrics


# ---------- Intersection stagger -----------------------------------------
def run_intersection_stagger(stagger, base_params, geom, T=20.0, TS=0.02,
                              tau_matrix=R_PLUS_90, c2_t=0.15, d_c=70.0,
                              N_per_flock=4, lane_offset=3.5, a_max=9.0):
    """E+W flocks start at dist_back=60m, N+S flocks start at 60+stagger."""
    from control_alpha import control_alpha
    from control_gamma import control_gamma
    from control_tau import control_tau
    from control_beta_cross import control_beta_cross

    params = dict(base_params)
    params['tau_matrix'] = tau_matrix
    params['c1_t'] = 0.0
    params['c2_t'] = c2_t
    params['d_c'] = d_c
    params.pop('predict_suppress_threshold', None)

    flock_inits = []
    p_d_dict = {}
    for k, (d, dist_back) in enumerate([
        ('E', 60.0), ('W', 60.0),
        ('N', 60.0 + stagger), ('S', 60.0 + stagger),
    ], start=1):
        q, p = make_cross_flock(d, N_per_flock, params, geom, dist_back, lane_offset)
        flock_inits.append((q, p))
        p_d_dict[k] = p[0].copy()
    params['p_d_per_flock'] = p_d_dict

    num_steps = int(round(T / TS)) + 1
    N_total = sum(q.shape[0] for q, _ in flock_inits)
    q_arr = np.zeros((N_total, 2, num_steps))
    p_arr = np.zeros((N_total, 2, num_steps))
    flock_id = np.zeros(N_total, dtype=int)
    offset = 0
    for k, (qi, pi) in enumerate(flock_inits, start=1):
        n = qi.shape[0]
        q_arr[offset:offset+n, :, 0] = qi
        p_arr[offset:offset+n, :, 0] = pi
        flock_id[offset:offset+n] = k
        offset += n

    for t in range(num_steps - 1):
        qt, pt = q_arr[:,:,t], p_arr[:,:,t]
        ut = np.zeros((N_total, 2))
        for i in range(N_total):
            ut[i] = (control_alpha(i, qt, pt, flock_id, params)
                     + control_beta_cross(i, qt, pt, geom, params)
                     + control_gamma(i, qt, pt, flock_id, params)
                     + control_tau  (i, qt, pt, flock_id, params))
        for i in range(N_total):
            mag = np.linalg.norm(ut[i])
            if mag > a_max:
                ut[i] = a_max * ut[i] / mag
        p_arr[:,:,t+1] = pt + TS * ut
        q_arr[:,:,t+1] = qt + TS * p_arr[:,:,t+1]
    return q_arr, p_arr, flock_id


def main_intersection():
    base = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,
    }
    geom = {'half_road': 7.0, 'half_inter': 7.0}

    staggers = [0, 5, 10, 15, 20, 30, 50, 80]
    rows = []
    for stagger in staggers:
        q, p, fid = run_intersection_stagger(stagger, base, geom, T=22.0)
        inter, off_road, _ = cross_metrics(q, p, fid, geom)
        rows.append((stagger, inter, off_road))

    print('=== INTERSECTION stagger sweep (N+S delayed by `stagger` m) ===')
    print(f'{"stagger":>10}  {"inter":>7}  {"off-road":>9}')
    for s, im, off in rows:
        flag = '★' if im >= 2.0 and off == 0 else ('safe' if im >= 2.0 else 'unsafe')
        print(f'{s:10.1f}m {im:6.2f}m  {off:8d}  {flag}')
    return rows


def main_merge():
    base = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,'d_c':40.0,'c1_t':0.0,'c2_t':0.08,
    }
    geom = {'L_merge': 30.0, 'main_top': 7.0, 'main_bot': 0.0, 'ramp_bot': -7.0}

    staggers = [0, 3, 6, 9, 12, 15, 20, 30, 50]
    rows = []
    TS = 0.02
    for stagger in staggers:
        q, p, fid = run_asym_merge(4, 4, base, geom,
                                    ramp_stagger=stagger, T=18.0, TS=TS,
                                    lead_x_main=-40.0)
        pmin, off, in_main, stalled = merge_metrics(q, geom, 4, 4, TS)
        rows.append((stagger, pmin, off, stalled))

    print('\n=== MERGE stagger sweep (on-ramp delayed by `stagger` m) ===')
    print(f'{"stagger":>10}  {"pair":>7}  {"off-road":>9}  {"stalled":>7}')
    for s, pm, off, st in rows:
        flag = '★' if pm >= 2.0 else 'unsafe'
        print(f'{s:10.1f}m {pm:6.2f}m  {off:8d}  {st:7d}  {flag}')
    return rows


def main():
    rows_inter = main_intersection()
    rows_merge = main_merge()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    xs = [r[0] for r in rows_inter]; ys = [r[1] for r in rows_inter]
    off = [r[2] for r in rows_inter]
    ax.plot(xs, ys, '-o', color='C0', label='inter-flock min')
    ax.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='car width 2 m')
    ax.set_xlabel('N+S stagger [m]'); ax.set_ylabel('inter-flock min [m]')
    ax.set_title('Intersection stagger sweep'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.bar(xs, off, width=2.0, alpha=0.3, color='crimson', label='off-road')
    ax2.set_ylabel('off-road (agent·step)', color='crimson')

    ax = axes[1]
    xs = [r[0] for r in rows_merge]; ys = [r[1] for r in rows_merge]
    ax.plot(xs, ys, '-o', color='C2')
    ax.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='car width 2 m')
    ax.set_xlabel('on-ramp stagger [m]'); ax.set_ylabel('pair-min [m]')
    ax.set_title('Merge stagger sweep'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Stagger sweeps — when does external timing make the algorithm safe?')
    fig.tight_layout()
    fig.savefig('sweep_stagger.png', dpi=100, bbox_inches='tight')
    print('\nsaved sweep_stagger.png')


if __name__ == '__main__':
    main()
