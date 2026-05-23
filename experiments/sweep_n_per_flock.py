"""Sweep flock size (cars per flock) and measure crossing behavior.

Hypothesis: denser flocks => more simultaneous crossing pairs => more
geometric interference => inter-flock min degrades.

Initial geometry: each flock is a single-row train of N cars spaced by d_a.
The innermost car (closest to x=0) is anchored at x = +/- 20 across all
sweeps so the tau-engagement window is comparable across flock sizes.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.control_alpha import control_alpha
from flocking_lib.control_beta import control_beta
from flocking_lib.control_gamma import control_gamma
from flocking_lib.control_tau import control_tau


def run_sim(N_per_flock, params, scenario):
    N           = 2 * N_per_flock
    TS          = scenario['TS']
    T           = scenario['T']
    num_steps   = int(round(T / TS)) + 1
    a_max       = scenario['a_max']
    y_lo, y_hi  = 0.0, scenario['y_hi']
    y_mid       = 0.5 * (y_lo + y_hi)

    flock_id = np.concatenate([np.ones(N_per_flock, dtype=int),
                               2 * np.ones(N_per_flock, dtype=int)])
    q = np.zeros((N, 2, num_steps))
    p = np.zeros((N, 2, num_steps))

    # Innermost car of each flock at x = -/+ x_inner.
    x_inner = scenario['x_inner']
    d_a = params['d_a']
    flock1_x = np.linspace(-x_inner - (N_per_flock - 1) * d_a, -x_inner, N_per_flock)
    flock2_x = np.linspace( x_inner,  x_inner + (N_per_flock - 1) * d_a, N_per_flock)
    q[:N_per_flock,  0, 0] = flock1_x
    q[ N_per_flock:, 0, 0] = flock2_x
    q[:, 1, 0] = y_mid
    p[:N_per_flock,  :, 0] = np.tile(params['p_d_flock1'], (N_per_flock, 1))
    p[ N_per_flock:, :, 0] = np.tile(params['p_d_flock2'], (N_per_flock, 1))

    for t in range(num_steps - 1):
        qt, pt = q[:, :, t], p[:, :, t]
        ut = np.zeros((N, 2))
        for i in range(N):
            ut[i] = (control_alpha(i, qt, pt, flock_id, params)
                     + control_beta (i, qt, pt, y_lo, y_hi, params)
                     + control_gamma(i, qt, pt, flock_id, params)
                     + control_tau  (i, qt, pt, flock_id, params))
        for i in range(N):
            mag = np.linalg.norm(ut[i])
            if mag > a_max:
                ut[i] = a_max * ut[i] / mag
        p[:, :, t + 1] = pt + TS * ut
        q[:, :, t + 1] = qt + TS * p[:, :, t + 1]

    return q, p, flock_id


def metrics(q, flock_id, y_hi, d_b):
    N = q.shape[0]
    N1 = (flock_id == 1).sum()
    inter_min = np.inf
    intra_min = np.inf
    for t in range(q.shape[2]):
        # inter-flock
        for i in range(N1):
            for j in range(N1, N):
                d = np.linalg.norm(q[i, :, t] - q[j, :, t])
                if d < inter_min:
                    inter_min = d
        # intra-flock
        for group in (range(N1), range(N1, N)):
            ag = list(group)
            for ii, i in enumerate(ag):
                for j in ag[ii + 1:]:
                    d = np.linalg.norm(q[i, :, t] - q[j, :, t])
                    if d < intra_min:
                        intra_min = d
    near_wall = ((q[:, 1, :] < d_b) | (q[:, 1, :] > y_hi - d_b))
    wall_proximity_fraction = near_wall.mean()
    escapes = int(((q[:, 1, :] < 0) | (q[:, 1, :] > y_hi)).sum())

    # clearance at x-encounter (when flock centroids share an x)
    x1 = q[:N1, 0, :].mean(axis=0)
    x2 = q[N1:, 0, :].mean(axis=0)
    cross_idx = int(np.argmin(np.abs(x1 - x2)))
    y1 = q[:N1, 1, cross_idx].mean()
    y2 = q[N1:, 1, cross_idx].mean()
    clearance = abs(y2 - y1)

    return dict(inter_min=inter_min, intra_min=intra_min,
                wall_proximity_fraction=wall_proximity_fraction,
                escapes=escapes, clearance_at_x0=clearance)


def main():
    params = {
        'e':    0.1,
        'a':    5, 'b': 5,
        'd_a':  7,  'r_a': 1.2 * 7, 'h_a': 0.2,
        'c1_a': 5,  'c2_a': 2 * np.sqrt(5),
        'd_b':  3.0, 'h_b': 0.2,
        'c1_b': 200, 'c2_b': 2 * np.sqrt(200),
        'c_g':  1.5,
        'd_c':  30.0, 'c1_t': 0.02, 'c2_t': 0.08,
        'p_d_flock1': np.array([+10.0, 0.0]),
        'p_d_flock2': np.array([-10.0, 0.0]),
    }
    scenario = dict(
        TS=0.02, T=18, a_max=9.0,
        x_inner=20.0,    # innermost car of each flock at x = +/- 20
        y_hi=14.4,
    )

    Ns = [2, 3, 4, 6, 8, 10]
    rows = []
    for N_pf in Ns:
        q, p, flock_id = run_sim(N_pf, params, scenario)
        m = metrics(q, flock_id, scenario['y_hi'], params['d_b'])
        rows.append((N_pf, m))
        print(f'N_per_flock={N_pf:2d}  inter={m["inter_min"]:5.2f}m  '
              f'intra={m["intra_min"]:5.2f}m  wall_frac={m["wall_proximity_fraction"]:.2f}  '
              f'escapes={m["escapes"]:4d}  clearance@x0={m["clearance_at_x0"]:5.2f}m')

    # Plot
    Ns_arr = [r[0] for r in rows]
    inter = [r[1]['inter_min'] for r in rows]
    intra = [r[1]['intra_min'] for r in rows]
    clear0 = [r[1]['clearance_at_x0'] for r in rows]
    wall = [r[1]['wall_proximity_fraction'] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(Ns_arr, inter, '-o', label='inter-flock min')
    axes[0].plot(Ns_arr, intra, '-s', label='intra-flock min')
    axes[0].plot(Ns_arr, clear0, '-^', label='clearance at x-encounter')
    axes[0].axhline(2.0, color='r', linestyle='--', alpha=0.4, label='car width 2 m')
    axes[0].set_xlabel('cars per flock')
    axes[0].set_ylabel('distance [m]')
    axes[0].set_title('Separation vs flock size')
    axes[0].legend(fontsize=8); axes[0].grid(True)

    axes[1].plot(Ns_arr, wall, '-o', color='purple')
    axes[1].set_xlabel('cars per flock')
    axes[1].set_ylabel('fraction of (agent x step)')
    axes[1].set_title(r'Time spent within $d_b$ of wall')
    axes[1].grid(True)

    fig.suptitle('McKenzie τ-agent: flock-size sweep (y_hi=14.4 m)')
    fig.tight_layout()
    fig.savefig('sweep_n_per_flock.png', dpi=110, bbox_inches='tight')
    print('saved sweep_n_per_flock.png')


if __name__ == '__main__':
    main()
