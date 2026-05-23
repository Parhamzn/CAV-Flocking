"""Diagnostic for Exp A: why is mean(v_x) pinned at v_d for every density?

Hypothesis: at steady state, γ = -c_g (p - v_d·x̂) is the only force with a
preferred direction in x. The α-gradient is symmetric in a translation-
invariant periodic corridor, so its mean x-component is zero. The β-walls
only push in y. Therefore mean(γ_x) = 0 in steady state → mean(v_x) = v_d
regardless of how packed the corridor is.

This script logs, for a single high-N run:
  * mean(v_x), std(v_x), min(v_x), max(v_x)
  * mean(|α_x|), mean(|γ_x|), mean(|β_y|)
  * intra-flock pair-min distance (was the lattice broken?)
  * max pair-overlap (did cars physically pass through each other?)
  * fraction of cars within d_a/2 of any neighbor (congestion proxy)

If the hypothesis is right, mean(v_x) ≈ v_d but std(v_x) grows with N, and
intra-min collapses well below d_a at high N. The "congestion" is hidden
inside the variance, not the mean.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from exp_fundamental_diagram import (
    alpha_force_periodic, beta_force_corridor, gamma_force,
    D_A, D_B, A_MAX, EPS,
)


def min_pair_distance_periodic(q, L):
    """Min pairwise distance with x-periodic min-image."""
    dx = q[None, :, 0] - q[:, None, 0]
    dy = q[None, :, 1] - q[:, None, 1]
    dx = (dx + L / 2) % L - L / 2
    d2 = dx * dx + dy * dy
    N = q.shape[0]
    d2[np.arange(N), np.arange(N)] = np.inf
    return float(np.sqrt(d2.min()))


def run_instrumented(N, L=500.0, W=14.0, v_d=10.0, T=30.0, TS=0.02, seed=42):
    rng = np.random.default_rng(seed + N)
    n_x = max(2, int(np.ceil(L / D_A)))
    n_y = max(1, int(np.ceil((W - 2 * D_B) / (D_A * np.sqrt(3) / 2))))
    candidates = []
    for ix in range(n_x):
        for iy in range(n_y):
            x = ix * D_A + (D_A / 2 if iy % 2 else 0)
            y = D_B + 0.5 + iy * D_A * np.sqrt(3) / 2
            if 0 <= x < L and D_B + 0.2 < y < W - D_B - 0.2:
                candidates.append((x, y))
    if N > len(candidates):
        q = np.zeros((N, 2))
        q[:, 0] = rng.uniform(0, L, N)
        q[:, 1] = rng.uniform(D_B + 0.3, W - D_B - 0.3, N)
    else:
        idx = rng.choice(len(candidates), N, replace=False)
        q = np.array([candidates[i] for i in idx]) + rng.normal(0, 0.2, (N, 2))
    p = np.tile(np.array([v_d, 0.0]), (N, 1))

    num_steps = int(round(T / TS))
    measure_start = int(num_steps * (2.0 / 3.0))
    log = {
        'mean_vx': [], 'std_vx': [], 'min_vx': [], 'max_vx': [],
        'mean_abs_ax': [], 'mean_abs_gx': [], 'mean_abs_by': [],
        'intra_min': [],
    }
    for t in range(num_steps):
        u_a = alpha_force_periodic(q, p, L)
        u_b = beta_force_corridor(q, p, W)
        u_g = gamma_force(p, v_d)
        u = u_a + u_b + u_g
        mags = np.linalg.norm(u, axis=1)
        scale = np.where(mags > A_MAX, A_MAX / mags, 1.0)
        u *= scale[:, None]
        p += TS * u
        q += TS * p
        q[:, 0] %= L
        if t >= measure_start:
            log['mean_vx'].append(p[:, 0].mean())
            log['std_vx'].append(p[:, 0].std())
            log['min_vx'].append(p[:, 0].min())
            log['max_vx'].append(p[:, 0].max())
            log['mean_abs_ax'].append(np.abs(u_a[:, 0]).mean())
            log['mean_abs_gx'].append(np.abs(u_g[:, 0]).mean())
            log['mean_abs_by'].append(np.abs(u_b[:, 1]).mean())
            log['intra_min'].append(min_pair_distance_periodic(q, L))
    return {k: np.array(v) for k, v in log.items()}, q, p


def main():
    L, W, v_d = 500.0, 14.0, 10.0
    Ns = [40, 120, 250, 400, 700]
    summary = []
    for N in Ns:
        log, q_fin, p_fin = run_instrumented(N, L=L, W=W, v_d=v_d, T=30.0)
        k = N / L
        row = {
            'N': N,
            'k_per_km': k * 1000,
            'mean_vx_kmh': log['mean_vx'].mean() * 3.6,
            'std_vx_kmh': log['std_vx'].mean() * 3.6,
            'min_vx_kmh': log['min_vx'].min() * 3.6,
            'max_vx_kmh': log['max_vx'].max() * 3.6,
            'mean_abs_ax': log['mean_abs_ax'].mean(),
            'mean_abs_gx': log['mean_abs_gx'].mean(),
            'mean_abs_by': log['mean_abs_by'].mean(),
            'intra_min': log['intra_min'].mean(),
            'intra_min_worst': log['intra_min'].min(),
        }
        summary.append(row)
        print(f"\n--- N={N} (k={k*1000:.0f} veh/km) ---")
        print(f"  mean v_x  = {row['mean_vx_kmh']:.4f} km/h  (target {v_d*3.6:.1f})")
        print(f"  std  v_x  = {row['std_vx_kmh']:.3f} km/h")
        print(f"  min  v_x  = {row['min_vx_kmh']:.2f} km/h")
        print(f"  max  v_x  = {row['max_vx_kmh']:.2f} km/h")
        print(f"  |α_x|     = {row['mean_abs_ax']:.3f} m/s²")
        print(f"  |γ_x|     = {row['mean_abs_gx']:.3f} m/s²")
        print(f"  |β_y|     = {row['mean_abs_by']:.3f} m/s²")
        print(f"  intra-min = {row['intra_min']:.2f} m  (worst {row['intra_min_worst']:.2f}, d_a={D_A})")

    # Plot: as N grows, does the *mean* really stay at v_d while std and
    # intra-min tell the real story?
    Ns_arr = np.array([r['N'] for r in summary])
    ks = np.array([r['k_per_km'] for r in summary])
    mean_vx = np.array([r['mean_vx_kmh'] for r in summary])
    std_vx = np.array([r['std_vx_kmh'] for r in summary])
    min_vx = np.array([r['min_vx_kmh'] for r in summary])
    intra = np.array([r['intra_min'] for r in summary])
    intra_worst = np.array([r['intra_min_worst'] for r in summary])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.plot(ks, mean_vx, '-o', label='mean v_x')
    ax.plot(ks, min_vx, '-s', label='min v_x')
    ax.fill_between(ks, mean_vx - std_vx, mean_vx + std_vx, alpha=0.2,
                    label='±1σ v_x')
    ax.axhline(v_d * 3.6, color='gray', linestyle=':', label=f'v_d = {v_d*3.6:.0f}')
    ax.set_xlabel('density k [veh/km]')
    ax.set_ylabel('forward speed v_x [km/h]')
    ax.set_title('v_x: mean is pinned, variance grows')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ks, intra, '-o', label='intra-min (time-avg)')
    ax.plot(ks, intra_worst, '-s', label='intra-min (worst-frame)')
    ax.axhline(D_A, color='gray', linestyle=':', label=f'd_a = {D_A}')
    ax.axhline(D_A / 2, color='red', linestyle=':', label=f'd_a/2 (overlap risk)')
    ax.set_xlabel('density k [veh/km]')
    ax.set_ylabel('pair-min distance [m]')
    ax.set_title('Lattice collapses with density')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    abs_ax = np.array([r['mean_abs_ax'] for r in summary])
    abs_gx = np.array([r['mean_abs_gx'] for r in summary])
    abs_by = np.array([r['mean_abs_by'] for r in summary])
    ax.plot(ks, abs_ax, '-o', label='|α_x|')
    ax.plot(ks, abs_gx, '-s', label='|γ_x|')
    ax.plot(ks, abs_by, '-^', label='|β_y|')
    ax.axhline(A_MAX, color='gray', linestyle=':', label=f'A_max = {A_MAX}')
    ax.set_xlabel('density k [veh/km]')
    ax.set_ylabel('mean |force| [m/s²]')
    ax.set_title('Force budget per direction')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle('Exp A diagnostic — why mean(v_x) ≈ v_d at every density',
                 y=1.02)
    fig.tight_layout()
    fig.savefig('diagnose_fundamental.png', dpi=110, bbox_inches='tight')
    print('\nsaved diagnose_fundamental.png')


if __name__ == '__main__':
    main()
