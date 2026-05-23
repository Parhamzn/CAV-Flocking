"""Experiment B: smoothness — flocking vs. lane-locked baseline.

Same periodic corridor as Exp A (L=500 m, W=14 m, v_d=10 m/s). Compare:

  flocking   : full α + β + γ dynamics (2D lateral motion allowed).
  lane-locked: same forces applied, then y-component of acceleration and
               velocity is zeroed each step. Each car is pinned to its
               initial y. β still computed but its only output is y-force,
               so it contributes nothing to the locked motion.

Both conditions share the SAME initial random positions and seed, so
differences in metrics reflect algorithmic behaviour, not initial state.

Initial positions are uniform random (not pre-relaxed to a hex lattice),
so the algorithm must do real organisational work during the run — that
is what we measure.

Per-car metrics over the last 20 s of a 30 s run:
  rms_ax  : sqrt(mean(a_x²))                — longitudinal smoothness
  rms_ay  : sqrt(mean(a_y²))                — lateral activity
  peak_jx : max |Δa_x / Δt|                 — longitudinal jerk peak
  peak_jy : max |Δa_y / Δt|                 — lateral jerk peak
  std_vx  : std(v_x)                        — speed wobble

Each per-car array is aggregated across cars as (mean, p95).

Lane-locked predictions: rms_ay = peak_jy = 0 exactly. Other metrics
should be comparable. The interesting question is which condition has
lower longitudinal forces — does lateral freedom relieve x-load, or
does it add coupling and make x worse?
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from exp_fundamental_diagram import (
    alpha_force_periodic, beta_force_corridor, gamma_force,
    D_A, D_B, A_MAX,
)


def run_corridor_smoothness(N, L, W, v_d, lane_locked, T=30.0, TS=0.02, seed=42):
    """Returns per-car metrics over the measurement window.
    Same initial state for matching (seed, N, L, W, v_d, T, TS) regardless
    of lane_locked — the only difference is whether y motion is allowed.
    """
    rng = np.random.default_rng(seed + N)
    q = np.zeros((N, 2))
    q[:, 0] = rng.uniform(0, L, N)
    q[:, 1] = rng.uniform(D_B + 0.5, W - D_B - 0.5, N)
    p = np.tile(np.array([v_d, 0.0]), (N, 1))

    num_steps = int(round(T / TS))
    measure_start = int(num_steps * (1.0 / 3.0))  # discard first 10 s of settling
    M = num_steps - measure_start

    ax_hist = np.zeros((M, N))
    ay_hist = np.zeros((M, N))
    vx_hist = np.zeros((M, N))

    prev_ax = np.zeros(N)
    prev_ay = np.zeros(N)
    peak_jx = np.zeros(N)
    peak_jy = np.zeros(N)
    first_jerk_frame = True

    for t in range(num_steps):
        u = alpha_force_periodic(q, p, L)
        u += beta_force_corridor(q, p, W)
        u += gamma_force(p, v_d)
        mags = np.linalg.norm(u, axis=1)
        scale = np.where(mags > A_MAX, A_MAX / np.maximum(mags, 1e-9), 1.0)
        u *= scale[:, None]
        if lane_locked:
            u[:, 1] = 0.0
            p[:, 1] = 0.0
        p += TS * u
        q += TS * p
        q[:, 0] %= L

        if t >= measure_start:
            idx = t - measure_start
            ax_hist[idx] = u[:, 0]
            ay_hist[idx] = u[:, 1]
            vx_hist[idx] = p[:, 0]
            if first_jerk_frame:
                first_jerk_frame = False
            else:
                jx = np.abs(u[:, 0] - prev_ax) / TS
                jy = np.abs(u[:, 1] - prev_ay) / TS
                peak_jx = np.maximum(peak_jx, jx)
                peak_jy = np.maximum(peak_jy, jy)
            prev_ax = u[:, 0].copy()
            prev_ay = u[:, 1].copy()

    rms_ax = np.sqrt((ax_hist ** 2).mean(axis=0))
    rms_ay = np.sqrt((ay_hist ** 2).mean(axis=0))
    std_vx = vx_hist.std(axis=0)
    return {
        'rms_ax': rms_ax, 'rms_ay': rms_ay,
        'peak_jx': peak_jx, 'peak_jy': peak_jy,
        'std_vx': std_vx,
    }


def aggregate(per_car):
    return {k: (float(v.mean()), float(np.percentile(v, 95)))
            for k, v in per_car.items()}


def main():
    L, W, v_d = 500.0, 14.0, 10.0
    N_vals = [30, 60, 90, 120, 140, 160]

    print(f'L={L} m, W={W} m, v_d={v_d} m/s')
    print(f'{"N":>4}  {"cond":>11}  '
          f'{"rms_ax":>15}  {"rms_ay":>15}  '
          f'{"peak_jx":>15}  {"peak_jy":>15}  '
          f'{"std_vx":>15}')
    results = {'flocking': [], 'lane_locked': []}
    for N in N_vals:
        for cond, locked in [('flocking', False), ('lane_locked', True)]:
            per_car = run_corridor_smoothness(N, L, W, v_d, lane_locked=locked)
            agg = aggregate(per_car)
            results[cond].append((N, agg))
            print(f'{N:4d}  {cond:>11}  '
                  f'{agg["rms_ax"][0]:6.3f} / {agg["rms_ax"][1]:5.2f}p95  '
                  f'{agg["rms_ay"][0]:6.3f} / {agg["rms_ay"][1]:5.2f}p95  '
                  f'{agg["peak_jx"][0]:6.1f} / {agg["peak_jx"][1]:5.1f}p95  '
                  f'{agg["peak_jy"][0]:6.1f} / {agg["peak_jy"][1]:5.1f}p95  '
                  f'{agg["std_vx"][0]:6.3f} / {agg["std_vx"][1]:5.2f}p95')

    # plot mean and p95 for each metric, both conditions, across N
    metric_keys = ['rms_ax', 'rms_ay', 'peak_jx', 'peak_jy', 'std_vx']
    titles = {
        'rms_ax': 'RMS a_x  [m/s²] — longitudinal smoothness',
        'rms_ay': 'RMS a_y  [m/s²] — lateral activity',
        'peak_jx': 'Peak |jerk_x|  [m/s³]',
        'peak_jy': 'Peak |jerk_y|  [m/s³]',
        'std_vx': 'std(v_x)  [m/s] — speed wobble',
    }
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()
    Ns = np.array(N_vals)
    for ax, key in zip(axes, metric_keys):
        for cond, color, marker in [('flocking', 'C0', 'o'),
                                    ('lane_locked', 'C3', 's')]:
            means = np.array([r[1][key][0] for r in results[cond]])
            p95s = np.array([r[1][key][1] for r in results[cond]])
            ax.plot(Ns, means, color=color, marker=marker,
                    label=f'{cond} (mean)')
            ax.plot(Ns, p95s, color=color, marker=marker, linestyle='--',
                    alpha=0.6, label=f'{cond} (p95)')
        ax.set_xlabel('N (cars in corridor)')
        ax.set_ylabel(key)
        ax.set_title(titles[key])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # 6th panel: summary ratios (flocking / lane-locked) for x-only metrics
    ax = axes[5]
    ratios = {}
    for key in ['rms_ax', 'peak_jx', 'std_vx']:
        flock = np.array([r[1][key][0] for r in results['flocking']])
        locked = np.array([r[1][key][0] for r in results['lane_locked']])
        ratios[key] = flock / np.maximum(locked, 1e-9)
    for key, color in [('rms_ax', 'C2'), ('peak_jx', 'C4'), ('std_vx', 'C5')]:
        ax.plot(Ns, ratios[key], '-o', color=color,
                label=f'{key}: flocking / lane-locked')
    ax.axhline(1.0, color='gray', linestyle=':', label='equal')
    ax.set_xlabel('N')
    ax.set_ylabel('ratio (flocking / lane-locked)')
    ax.set_title('x-axis force ratio: does lateral freedom help or hurt?')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle('Exp B — smoothness comparison: flocking vs. lane-locked',
                 y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig('exp_smoothness.png', dpi=110, bbox_inches='tight')
    print('\nsaved exp_smoothness.png')


if __name__ == '__main__':
    main()
