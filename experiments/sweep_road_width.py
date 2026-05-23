"""Sweep road width and measure how the alpha+beta+gamma+tau flocking algorithm
behaves on roads of different y-extent.

The McKenzie 2012 paper was set in open 2D space; the IVT project applies it to
a bounded freeway. This sweep explores the resulting tau/beta tension.

Metrics per road width:
  - inter_min:                smallest distance between any flock-1/flock-2 pair
  - intra_min:                smallest within-flock distance (alpha-lattice fidelity)
  - wall_proximity_fraction:  fraction of (agent x timestep) entries within d_b of a wall
  - escapes:                  count of timesteps where any agent left the lane
  - clearance_at_x0:          y-gap between flock centroids at the moment of x-encounter
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.control_alpha import control_alpha
from flocking_lib.control_beta import control_beta
from flocking_lib.control_gamma import control_gamma
from flocking_lib.control_tau import control_tau


def run_sim(y_hi, params, scenario):
    """Run a single simulation, return (q, p, u, flock_id)."""
    N           = scenario['N']
    N_per_flock = N // 2
    TS          = scenario['TS']
    T           = scenario['T']
    num_steps   = int(round(T / TS)) + 1
    a_max       = scenario['a_max']
    y_lo        = 0.0
    y_mid       = 0.5 * (y_lo + y_hi)

    flock_id = np.concatenate([np.ones(N_per_flock, dtype=int),
                               2 * np.ones(N_per_flock, dtype=int)])

    q = np.zeros((N, 2, num_steps))
    p = np.zeros((N, 2, num_steps))
    u = np.zeros((N, 2, num_steps))

    flock1_x = np.linspace(scenario['x_start'],
                           scenario['x_start'] + (N_per_flock - 1) * params['d_a'],
                           N_per_flock)
    flock2_x = np.linspace(-scenario['x_start'] - (N_per_flock - 1) * params['d_a'],
                           -scenario['x_start'],
                           N_per_flock)
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
        u[:, :, t]     = ut
        p[:, :, t + 1] = pt + TS * ut
        q[:, :, t + 1] = qt + TS * p[:, :, t + 1]

    return q, p, u, flock_id


def metrics(q, p, flock_id, y_hi, d_b, TS):
    N = q.shape[0]
    N1 = (flock_id == 1).sum()
    inter_min = np.inf
    intra_min = np.inf
    for t in range(q.shape[2]):
        for i in range(N1):
            for j in range(N1, N):
                d = np.linalg.norm(q[i, :, t] - q[j, :, t])
                if d < inter_min:
                    inter_min = d
        for group in (range(N1), range(N1, N)):
            ag = list(group)
            for ii, i in enumerate(ag):
                for j in ag[ii + 1:]:
                    d = np.linalg.norm(q[i, :, t] - q[j, :, t])
                    if d < intra_min:
                        intra_min = d
    # wall proximity: fraction of (agent, step) entries inside beta-zone
    near_wall = ((q[:, 1, :] < d_b) | (q[:, 1, :] > y_hi - d_b))
    wall_proximity_fraction = near_wall.mean()
    escapes = int(((q[:, 1, :] < 0) | (q[:, 1, :] > y_hi)).sum())

    # y-gap at moment of x-encounter (flock centroids share an x)
    x1_centroid = q[:N1, 0, :].mean(axis=0)
    x2_centroid = q[N1:, 0, :].mean(axis=0)
    crossing_idx = np.argmin(np.abs(x1_centroid - x2_centroid))
    y1 = q[:N1, 1, crossing_idx].mean()
    y2 = q[N1:, 1, crossing_idx].mean()
    clearance_at_x0 = abs(y2 - y1)

    return dict(
        inter_min=inter_min,
        intra_min=intra_min,
        wall_proximity_fraction=wall_proximity_fraction,
        escapes=escapes,
        clearance_at_x0=clearance_at_x0,
        t_inter_min=None,    # not tracking for brevity
    )


def main():
    # Fixed parameters (best McKenzie-corrected config)
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
    scenario = dict(N=8, TS=0.02, T=16, a_max=9.0, x_start=-80.0)

    widths = [10.0, 12.0, 14.4, 18.0, 24.0, 30.0]
    rows = []
    for y_hi in widths:
        q, p, u, flock_id = run_sim(y_hi, params, scenario)
        m = metrics(q, p, flock_id, y_hi, params['d_b'], scenario['TS'])
        rows.append((y_hi, m))
        print(f'y_hi={y_hi:5.1f}  inter={m["inter_min"]:5.2f}m  intra={m["intra_min"]:5.2f}m  '
              f'wall_frac={m["wall_proximity_fraction"]:.2f}  escapes={m["escapes"]:4d}  '
              f'clearance@x0={m["clearance_at_x0"]:5.2f}m')

    # Plot
    ys = [r[0] for r in rows]
    inter = [r[1]['inter_min'] for r in rows]
    intra = [r[1]['intra_min'] for r in rows]
    wall = [r[1]['wall_proximity_fraction'] for r in rows]
    clear0 = [r[1]['clearance_at_x0'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(ys, inter, '-o', label='inter-flock min')
    axes[0].plot(ys, intra, '-s', label='intra-flock min')
    axes[0].plot(ys, clear0, '-^', label='clearance at x-encounter')
    axes[0].axhline(2.0, color='r', linestyle='--', alpha=0.4, label='car width 2 m')
    axes[0].set_xlabel('road width y_hi [m]')
    axes[0].set_ylabel('distance [m]')
    axes[0].set_title('Separation vs road width')
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].plot(ys, wall, '-o', color='purple')
    axes[1].set_xlabel('road width y_hi [m]')
    axes[1].set_ylabel('fraction of (agent x step)')
    axes[1].set_title(r'Time spent within $d_b$ of either wall')
    axes[1].grid(True)

    escapes = [r[1]['escapes'] for r in rows]
    axes[2].bar(ys, escapes, width=1.2, color='crimson')
    axes[2].set_xlabel('road width y_hi [m]')
    axes[2].set_ylabel('escape (agent x step) count')
    axes[2].set_title('Lane-escape incidents')
    axes[2].grid(True, axis='y')

    fig.suptitle('McKenzie τ-agent: road-width sweep')
    fig.tight_layout()
    fig.savefig('sweep_road_width.png', dpi=110, bbox_inches='tight')
    print('saved sweep_road_width.png')


if __name__ == '__main__':
    main()
