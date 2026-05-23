"""Investigation #2: asymmetric flock-size sweep.

Fix flock 1 at N1=4, sweep N2 (number of cars in flock 2). Single-row
geometry both sides, y_hi=14.4 m road.

Metrics, per flock separately:
   * deflection magnitude        — |final_y - initial_y|
   * intra-flock min             — lattice integrity
   * wall_proximity fraction     — how much of the run the flock spent
                                   inside the beta zone
   * max y excursion             — peak distance from start (does it hit wall?)
And the inter-flock min.

Hypothesis: each car in flock i feels tau-force from each of N_j opposing
cars (within d_c). So per-car force on flock i ~ N_j. With Newton-like
intuition, deflection of flock i ~ N_j / N_i. Smaller flock deflects more.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.flock_layouts import grid_formation
from flocking_lib.multi_flock_sim import run_multi_flock


def per_flock_metrics(q, flock_id, y_hi, d_b):
    N1 = (flock_id == 1).sum()
    out = {}
    for fid in (1, 2):
        mask = flock_id == fid
        # deflection from initial y
        y0 = q[mask, 1, 0]
        y_final = q[mask, 1, -1]
        deflection = float(np.mean(y_final - y0))
        # intra-flock min over time
        idx = np.where(mask)[0]
        intra = np.inf
        for ii, i in enumerate(idx):
            for j in idx[ii + 1:]:
                d = np.linalg.norm(q[i, :, :] - q[j, :, :], axis=0).min()
                if d < intra:
                    intra = d
        near_wall = ((q[mask, 1, :] < d_b) | (q[mask, 1, :] > y_hi - d_b))
        wall_prox = near_wall.mean()
        max_excursion = float(np.max(np.abs(q[mask, 1, :] - 7.2)))
        out[f'flock{fid}'] = dict(deflection=deflection, intra_min=intra,
                                  wall_prox=wall_prox, max_excursion=max_excursion)
    # inter
    inter = np.inf
    for i in range(q.shape[0]):
        for j in range(i + 1, q.shape[0]):
            if flock_id[i] == flock_id[j]:
                continue
            d = np.linalg.norm(q[i, :, :] - q[j, :, :], axis=0).min()
            if d < inter:
                inter = d
    out['inter_min'] = float(inter)
    return out


def main():
    params = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,'d_c':40.0,'c1_t':0.0,'c2_t':0.08,
        'p_d_flock1': np.array([+10.0, 0.0]),
        'p_d_flock2': np.array([-10.0, 0.0]),
    }
    s = params['d_a']; v1 = params['p_d_flock1']; v2 = params['p_d_flock2']
    y_hi = 14.4; y_mid = 7.2
    N1 = 4

    N2_list = [1, 2, 3, 4, 6, 8, 10, 12]
    rows = []
    for N2 in N2_list:
        # Innermost car of each flock at x = ±20
        q1, p1 = grid_formation(1, N1,
                                x_center=-20 - (N1 - 1) * s / 2,
                                y_center=y_mid, spacing=s, vel=v1)
        q2, p2 = grid_formation(1, N2,
                                x_center=+20 + (N2 - 1) * s / 2,
                                y_center=y_mid, spacing=s, vel=v2)
        q, p, u, fid = run_multi_flock([(q1, p1), (q2, p2)], params,
                                        y_lo=0.0, y_hi=y_hi, T=14.0, TS=0.02)
        m = per_flock_metrics(q, fid, y_hi, params['d_b'])
        rows.append((N2, m))
        f1, f2 = m['flock1'], m['flock2']
        print(f'N1={N1} N2={N2:2d}  inter={m["inter_min"]:5.2f}m  '
              f'F1: defl={f1["deflection"]:+5.2f}m max_y={f1["max_excursion"]:.2f}m intra={f1["intra_min"]:.2f}m wall%={f1["wall_prox"]:.2f}  '
              f'F2: defl={f2["deflection"]:+5.2f}m max_y={f2["max_excursion"]:.2f}m intra={f2["intra_min"]:.2f}m wall%={f2["wall_prox"]:.2f}')

    # ---- Plots ------------------------------------------------------------
    N2s = [r[0] for r in rows]
    defl1 = [r[1]['flock1']['deflection'] for r in rows]
    defl2 = [r[1]['flock2']['deflection'] for r in rows]
    max1  = [r[1]['flock1']['max_excursion'] for r in rows]
    max2  = [r[1]['flock2']['max_excursion'] for r in rows]
    intra1 = [r[1]['flock1']['intra_min'] for r in rows]
    intra2 = [r[1]['flock2']['intra_min'] for r in rows]
    inter = [r[1]['inter_min'] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(N2s, defl1, '-o', label='Flock 1 (N1=4)')
    ax.plot(N2s, defl2, '-s', label='Flock 2 (variable N2)')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('N2 (cars in flock 2)'); ax.set_ylabel('mean final Δy [m]')
    ax.set_title('Net deflection per flock')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(N2s, max1, '-o', label='Flock 1 (N1=4)')
    ax.plot(N2s, max2, '-s', label='Flock 2 (variable N2)')
    ax.axhline(7.2 - params['d_b'], color='r', linestyle='--', alpha=0.4,
               label='reach β-zone edge')
    ax.set_xlabel('N2'); ax.set_ylabel('max |y - 7.2| during run [m]')
    ax.set_title('Peak y-excursion per flock')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(N2s, intra1, '-o', label='Flock 1 (N1=4)')
    ax.plot(N2s, intra2, '-s', label='Flock 2 (variable N2)')
    ax.axhline(params['d_a'], color='k', linestyle=':', alpha=0.5,
               label=f'd_a = {params["d_a"]} m')
    ax.set_xlabel('N2'); ax.set_ylabel('intra-flock min [m]')
    ax.set_title('α-lattice integrity per flock')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(N2s, inter, '-o', color='purple')
    ax.axhline(2.0, color='r', linestyle='--', alpha=0.4, label='car width')
    ax.set_xlabel('N2'); ax.set_ylabel('inter-flock min [m]')
    ax.set_title('Inter-flock minimum distance')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f'Asymmetric flock-size sweep (N1={N1}, y_hi=14.4 m, v_d=10)', y=0.995)
    fig.tight_layout()
    fig.savefig('sweep_asymmetric_size.png', dpi=110, bbox_inches='tight')
    print('saved sweep_asymmetric_size.png')


if __name__ == '__main__':
    main()
