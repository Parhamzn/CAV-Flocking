"""Systematic comparison of flock geometries.

Scenario matrix:
   A. Baseline:    1x4 vs 1x4 on 14.4 m road       (reference)
   B. Asym-small:  1x4 vs 1x2 on 14.4 m road       (one flock smaller)
   C. Asym-large:  1x4 vs 1x8 on 14.4 m road       (one flock bigger)
   D. 2D symmetric: 2x4 vs 2x4 on 24 m road        (multi-row)
   E. 2D asym:     2x4 vs 1x8 on 24 m road         (same size, different shape)
   F. Off-center:  1x4 (low) vs 1x4 (high) on 24 m (pre-sorted lanes)

For each: trajectory plot + inter-flock min, intra-flock min, clearance.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.flock_layouts import grid_formation
from flocking_lib.multi_flock_sim import run_multi_flock, encounter_metrics


def build_scenarios(d_a):
    s = d_a    # spacing for all formations
    v_d = 10.0
    v1 = np.array([+v_d, 0.0]); v2 = np.array([-v_d, 0.0])

    scenarios = {}

    # ---- A: baseline 1x4 vs 1x4 on 14.4 m road -----------------------------
    q1, p1 = grid_formation(1, 4, x_center=-20 - 1.5*s, y_center=7.2, spacing=s, vel=v1)
    q2, p2 = grid_formation(1, 4, x_center=+20 + 1.5*s, y_center=7.2, spacing=s, vel=v2)
    scenarios['A · baseline 1×4 vs 1×4'] = (q1, p1, q2, p2, 14.4)

    # ---- B: asym-small 1x4 vs 1x2 ------------------------------------------
    q1, p1 = grid_formation(1, 4, x_center=-20 - 1.5*s, y_center=7.2, spacing=s, vel=v1)
    q2, p2 = grid_formation(1, 2, x_center=+20 + 0.5*s, y_center=7.2, spacing=s, vel=v2)
    scenarios['B · asym 1×4 vs 1×2 (small foe)'] = (q1, p1, q2, p2, 14.4)

    # ---- C: asym-large 1x4 vs 1x8 ------------------------------------------
    q1, p1 = grid_formation(1, 4, x_center=-20 - 1.5*s, y_center=7.2, spacing=s, vel=v1)
    q2, p2 = grid_formation(1, 8, x_center=+20 + 3.5*s, y_center=7.2, spacing=s, vel=v2)
    scenarios['C · asym 1×4 vs 1×8 (big foe)'] = (q1, p1, q2, p2, 14.4)

    # ---- D: 2D symmetric 2x4 vs 2x4 on 24 m -------------------------------
    q1, p1 = grid_formation(2, 4, x_center=-20 - 1.5*s, y_center=12.0, spacing=s, vel=v1)
    q2, p2 = grid_formation(2, 4, x_center=+20 + 1.5*s, y_center=12.0, spacing=s, vel=v2)
    scenarios['D · 2×4 vs 2×4 (multi-row)'] = (q1, p1, q2, p2, 24.0)

    # ---- E: 2D asym 2x4 vs 1x8 on 24 m -----------------------------------
    q1, p1 = grid_formation(2, 4, x_center=-20 - 1.5*s, y_center=12.0, spacing=s, vel=v1)
    q2, p2 = grid_formation(1, 8, x_center=+20 + 3.5*s, y_center=12.0, spacing=s, vel=v2)
    scenarios['E · 2×4 vs 1×8 (same N, diff shape)'] = (q1, p1, q2, p2, 24.0)

    # ---- F: off-center 1x4 (low) vs 1x4 (high) on 24 m ---------------------
    q1, p1 = grid_formation(1, 4, x_center=-20 - 1.5*s, y_center=6.0,  spacing=s, vel=v1)
    q2, p2 = grid_formation(1, 4, x_center=+20 + 1.5*s, y_center=18.0, spacing=s, vel=v2)
    scenarios['F · off-center (pre-sorted lanes)'] = (q1, p1, q2, p2, 24.0)

    return scenarios


def main():
    params = {
        'e':0.1,'a':5,'b':5,
        'd_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,
        'd_c':40.0,'c1_t':0.0,'c2_t':0.08,
        'p_d_flock1': np.array([+10.0, 0.0]),
        'p_d_flock2': np.array([-10.0, 0.0]),
    }
    TS, T = 0.02, 12.0

    scenarios = build_scenarios(params['d_a'])

    # Run all scenarios
    results = []
    for label, (q1, p1, q2, p2, y_hi) in scenarios.items():
        q, p, u, fid = run_multi_flock(
            [(q1, p1), (q2, p2)], params, y_lo=0.0, y_hi=y_hi, T=T, TS=TS,
        )
        m = encounter_metrics(q, fid, y_hi, params['d_b'])
        results.append((label, q, p, fid, y_hi, m))
        print(f'{label:42s}  inter={m["inter_min"]:5.2f}m  '
              f'intra={m["intra_min"]:5.2f}m  wall%={m["wall_proximity_fraction"]:.2f}  '
              f'clearance={m["clearance_at_x0"]:5.2f}m')

    # 3x2 trajectory grid
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    for k, (label, q, p, fid, y_hi, m) in enumerate(results):
        ax = axes[k // 2, k % 2]
        N = q.shape[0]
        for i in range(N):
            xs, ys = q[i, 0, :], q[i, 1, :]
            col = 'b' if fid[i] == 1 else 'r'
            ax.plot(xs, ys, color=col, linewidth=1, alpha=0.7)
            ax.plot(xs[0],  ys[0],  marker='o', color=col, markersize=6)
            ax.plot(xs[-1], ys[-1], marker='s', color=col, markersize=6, fillstyle='none')
        ax.axhline(0,    color='k', linewidth=1.5)
        ax.axhline(y_hi, color='k', linewidth=1.5)
        ax.set_xlim(q[:,0,:].min()-5, q[:,0,:].max()+5)
        ax.set_ylim(-1, y_hi + 1)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title(f'{label}\n'
                     f'inter={m["inter_min"]:.2f}m  '
                     f'intra={m["intra_min"]:.2f}m  '
                     f'clearance={m["clearance_at_x0"]:.2f}m',
                     fontsize=9)
    fig.tight_layout()
    fig.savefig('exp_geometries.png', dpi=110, bbox_inches='tight')
    print('saved exp_geometries.png')


if __name__ == '__main__':
    main()
