"""Asymmetric flock configurations in the 4-way cross intersection.

Sweep a matrix of scenarios that vary which directions have flocks and
how big each flock is, then characterize the resulting behavior.

Scenarios:
  baseline       — 4-4-4-4 (already characterized; reference)
  big eastbound  — 8 vs 2-2-2 (one dominant flock)
  big horizontal — 8-8 vs 2-2 (axis asymmetry)
  3-way drop S   — 4-4-4-0 (one direction absent)
  2-way opposing — 4-0-0-4 (only one pair, perpendicular)
  2-way perp     — 4-4-0-0 (opposing horizontals, no vertical traffic)
  Single train   — 4-0-0-0 (just one flock; sanity check)
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.control_alpha import control_alpha
from flocking_lib.control_gamma import control_gamma
from flocking_lib.control_tau import control_tau, R_PLUS_90
from flocking_lib.control_beta_cross import control_beta_cross
from exp_cross_intersection import make_cross_flock, draw_road_geometry


def run_cross_asym(sizes_by_dir, base_params, geom, T=20.0, TS=0.02,
                   tau_matrix=R_PLUS_90, c2_t=0.15, d_c=70.0,
                   dist_back=70.0, lane_offset=3.5, a_max=9.0):
    """Run cross intersection with per-direction flock sizes.

    sizes_by_dir: dict mapping direction code ('E','W','N','S') to N (≥0).
    Zero means no flock from that direction.
    """
    params = dict(base_params)
    params['tau_matrix'] = tau_matrix
    params['c1_t'] = 0.0
    params['c2_t'] = c2_t
    params['d_c'] = d_c

    flock_inits = []
    p_d_dict = {}
    fid_counter = 1
    flock_labels = []
    for d in ['E', 'W', 'N', 'S']:
        N = sizes_by_dir.get(d, 0)
        if N == 0:
            continue
        q, p = make_cross_flock(d, N, params, geom, dist_back, lane_offset)
        flock_inits.append((q, p))
        p_d_dict[fid_counter] = p[0].copy()
        flock_labels.append((fid_counter, d))
        fid_counter += 1
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
    return q_arr, p_arr, flock_id, flock_labels


def metrics(q, p, flock_id, geom):
    """inter-flock min, off-road count, per-flock final direction."""
    N = q.shape[0]
    inter = np.inf
    for ti in range(q.shape[2]):
        for i in range(N):
            for j in range(i+1, N):
                if flock_id[i] == flock_id[j]: continue
                d = float(np.linalg.norm(q[i,:,ti] - q[j,:,ti]))
                if d < inter: inter = d
    off_road = sum(1 for ti in range(q.shape[2]) for i in range(N)
                   if abs(q[i,0,ti]) > geom['half_road']
                   and abs(q[i,1,ti]) > geom['half_road'])
    # Determine final direction by mean velocity over last 0.5s
    final = {}
    for fid in np.unique(flock_id):
        mask = flock_id == fid
        v_final = p[mask, :, -25:].mean(axis=(0, 2))
        speed = np.linalg.norm(v_final)
        if speed < 1.0:
            final[int(fid)] = 'stalled'
        else:
            dx, dy = v_final / speed
            if abs(dx) > abs(dy):
                final[int(fid)] = 'E' if dx > 0 else 'W'
            else:
                final[int(fid)] = 'N' if dy > 0 else 'S'
    return inter, off_road, final


def main():
    base = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,
    }
    geom = {'half_road': 7.0, 'half_inter': 7.0}

    scenarios = [
        ('A · baseline 4-4-4-4',         {'E':4,'W':4,'N':4,'S':4}),
        ('B · big eastbound 8-2-2-2',    {'E':8,'W':2,'N':2,'S':2}),
        ('C · big horizontal 8-8-2-2',   {'E':8,'W':8,'N':2,'S':2}),
        ('D · 3-way drop S',             {'E':4,'W':4,'N':4,'S':0}),
        ('E · 2-way opposing (E+W only)',{'E':4,'W':4,'N':0,'S':0}),
        ('F · 2-way perp (E+N only)',    {'E':4,'W':0,'N':4,'S':0}),
        ('G · single train E only',      {'E':4,'W':0,'N':0,'S':0}),
        ('H · diagonal mismatch (E+S)',  {'E':4,'W':0,'N':0,'S':4}),
    ]

    print(f'{"scenario":<35}  {"inter":>7}  {"off-road":>9}  {"in→out":>30}')
    print('-' * 90)
    results = []
    for label, sizes in scenarios:
        q, p, fid, labels = run_cross_asym(sizes, base, geom, T=20.0)
        inter, off_road, finals = metrics(q, p, fid, geom)
        flow_str = ' '.join(f'{d}→{finals[fk]}' for fk, d in labels)
        results.append((label, sizes, q, p, fid, labels, inter, off_road, finals))
        print(f'{label:<35}  {inter:6.2f}m  {off_road:8d}  {flow_str:>30}')

    # Render snapshot strips for selected scenarios
    flock_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple',
                    'tab:brown', 'tab:pink', 'tab:olive']
    TS = 0.02
    fig, axes = plt.subplots(len(results), 4, figsize=(13, 3*len(results)))
    if len(results) == 1:
        axes = axes[np.newaxis, :]
    for row, (label, sizes, q, p, fid, labels, inter, off_road, finals) in enumerate(results):
        strip_times = np.linspace(0, q.shape[2]-1, 4).astype(int)
        ext = 90
        for col, t in enumerate(strip_times):
            ax = axes[row, col]
            draw_road_geometry(ax, geom, ext)
            for i in range(q.shape[0]):
                c = flock_colors[(fid[i]-1) % len(flock_colors)]
                ax.scatter(q[i,0,t], q[i,1,t], c=c, s=25, edgecolors='k', linewidth=0.3)
            ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext)
            ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel(label, fontsize=8, weight='bold')
            ax.set_title(f't={t*TS:.1f}s', fontsize=8)
    fig.suptitle('Asymmetric flock configurations in 4-way cross intersection', fontsize=12)
    fig.tight_layout()
    fig.savefig('cross_asymmetric.png', dpi=80, bbox_inches='tight')
    print('\nsaved cross_asymmetric.png')


if __name__ == '__main__':
    main()
