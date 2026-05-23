"""Experiment E: lane formation — does the algorithm form emergent lanes?

The "lane-less" claim says cars spread freely across the y-axis. Strip-hex
theory says the opposite: on a W=14 m corridor with d_a=7 m and d_b=3 m
shoulders, exactly 2 hex rows fit at row spacing d_a·√3/2 ≈ 6.06 m. If the
α-lattice pulls cars into that arrangement from random initial conditions,
then the algorithm is lane-less by design but lane-forming in practice.

Setup
-----
Periodic corridor (L=500 m, W=14 m, v_d=10 m/s). Cars start at uniform-
random (x, y) inside the usable strip. Run 30 s. The first 10 s are
treated as transient; the y-distribution is sampled every TS in the last
20 s of the run and pooled into one histogram per N value.

Metrics
-------
* histogram of y over the measurement window (1.0 m bins inside the
  usable strip)
* Shannon entropy H = -Σ p log p, normalised to log(n_bins) so a perfectly
  uniform distribution scores 1.0 and a single-bin distribution scores 0.0
* mode count: number of local maxima in the smoothed histogram that exceed
  0.4 × the peak height — interpreted as the number of "informal lanes"
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from flocking_lib import multi_flock_sim  # noqa: F401  (registered in lib)

import exp_fundamental_diagram as A  # reuse vectorised α/β/γ + run helper


def _equilibrate_and_sample_y(N, L=500.0, W=14.0, v_d=10.0, T=30.0, TS=0.02,
                              seed=42):
    rng = np.random.default_rng(seed + N)
    q = np.zeros((N, 2))
    q[:, 0] = rng.uniform(0, L, N)
    q[:, 1] = rng.uniform(A.D_B + 0.3, W - A.D_B - 0.3, N)
    p = np.tile(np.array([v_d, 0.0]), (N, 1))
    num_steps = int(round(T / TS))
    measure_start = int(num_steps * (1.0 / 3.0))
    y_samples = []
    for t in range(num_steps):
        u = A.alpha_force_periodic(q, p, L)
        u += A.beta_force_corridor(q, p, W)
        u += A.gamma_force(p, v_d)
        mags = np.linalg.norm(u, axis=1)
        scale = np.where(mags > A.A_MAX, A.A_MAX / np.maximum(mags, 1e-9), 1.0)
        u *= scale[:, None]
        p += TS * u
        q += TS * p
        q[:, 0] %= L
        if t >= measure_start:
            y_samples.append(q[:, 1].copy())
    return np.concatenate(y_samples), q[:, 1].copy()


def entropy_and_modes(y_samples, W, bin_width=0.5):
    """Shannon entropy normalised to uniform, and mode count."""
    usable_lo = A.D_B
    usable_hi = W - A.D_B
    bins = np.arange(usable_lo, usable_hi + bin_width, bin_width)
    hist, edges = np.histogram(y_samples, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist / hist.sum() if hist.sum() > 0 else hist.astype(float)
    nonzero = p[p > 0]
    H = -np.sum(nonzero * np.log(nonzero))
    H_uniform = np.log(len(p))
    H_norm = H / H_uniform if H_uniform > 0 else 0.0
    smooth = np.convolve(hist, np.ones(3) / 3, mode='same')
    peak_idx, _ = find_peaks(smooth, height=0.4 * smooth.max(),
                             distance=int(2.0 / bin_width))
    return H_norm, len(peak_idx), centers, hist, peak_idx


def main():
    L, W, v_d = 500.0, 14.0, 10.0
    N_vals = [20, 40, 60, 90, 120, 140]
    bin_width = 0.5

    print(f'L={L} m, W={W} m, v_d={v_d} m/s, d_a={A.D_A}')
    print(f'strip-hex theory: 2 rows at y ≈ {A.D_B + 0.5:.1f} and y ≈ '
          f'{W - A.D_B - 0.5:.1f} (each spaced {A.D_A * np.sqrt(3)/2:.2f} m '
          f'apart vertically)\n')
    print(f'{"N":>4}  {"H/H_unif":>9}  {"#modes":>7}  {"interpretation":>40}')

    rows = []
    for N in N_vals:
        y_samples, y_final = _equilibrate_and_sample_y(N, L=L, W=W, v_d=v_d)
        H_norm, n_modes, centers, hist, peaks = entropy_and_modes(
            y_samples, W, bin_width=bin_width)
        if n_modes == 1:
            interp = "single band"
        elif n_modes == 2:
            interp = "TWO LANES emerge"
        elif n_modes == 3:
            interp = "three bands"
        else:
            interp = f"{n_modes} bands"
        rows.append(dict(N=N, H_norm=H_norm, n_modes=n_modes,
                         centers=centers, hist=hist, peaks=peaks,
                         y_final=y_final))
        print(f'{N:4d}  {H_norm:9.3f}  {n_modes:7d}  {interp:>40}')

    # ---- plot --------------------------------------------------------------
    n = len(rows)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(15, 8), squeeze=False)
    axes = axes.flatten()
    for ax, row in zip(axes, rows):
        ax.bar(row['centers'], row['hist'], width=bin_width * 0.95,
               color='C0', alpha=0.7, label='y-position frequency')
        if len(row['peaks']) > 0:
            ax.plot(row['centers'][row['peaks']],
                    row['hist'][row['peaks']], 'rv',
                    markersize=12, label=f"{row['n_modes']} modes")
        # mark expected hex-row positions
        for y_hex in [A.D_B + 0.5, W - A.D_B - 0.5]:
            ax.axvline(y_hex, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel('y [m]')
        ax.set_ylabel('count over measurement window')
        ax.set_title(f"N={row['N']} (k={row['N']/L*1000:.0f}/km)  "
                     f"H/H_unif={row['H_norm']:.2f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for k in range(len(rows), len(axes)):
        axes[k].axis('off')

    fig.suptitle('Exp E — y-distribution after settling from random initial '
                 'conditions\n(gray dotted lines = strip-hex theory row '
                 'centres)', y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig('exp_lane_formation.png', dpi=110, bbox_inches='tight')
    print('\nsaved exp_lane_formation.png')

    # ---- conclusion --------------------------------------------------------
    mode_counts = np.array([r['n_modes'] for r in rows])
    H_norms = np.array([r['H_norm'] for r in rows])
    print('\n=== Conclusion ===')
    if (mode_counts == 2).sum() >= len(rows) / 2:
        print('  CLAIM 3 ("lane-less") is FALSIFIED for typical densities.')
        print(f'  The algorithm spontaneously forms TWO emergent lanes from')
        print(f'  random initial conditions at most tested N values.')
    elif H_norms.mean() > 0.85:
        print('  CLAIM 3 supported: y-distribution remains close to uniform.')
    else:
        print('  Mixed result — see plot per N value.')
    print(f'  Mean normalised entropy across N: {H_norms.mean():.3f} '
          f'(1.0 = uniform, lower = more banded)')
    print(f'  Mode counts: {mode_counts.tolist()}')


if __name__ == '__main__':
    main()
