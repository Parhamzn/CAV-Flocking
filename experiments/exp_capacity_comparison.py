"""Experiment G: honest head-to-head capacity comparison.

Same 14 m corridor, sweep N for two conditions:
  lane_less : full α + β + γ on a 2-D corridor (this is Exp A reframed).
  lane_based: 2 fixed lanes at the strip-hex row centres, y locked.

For each N, run 30 s and measure (mean v_x, intra-min, std(v_x), drop
indicator). Find max-stable-density N* in each (intra-min ≥ d_a/2).

The hypothesis after Exp E is that BOTH conditions hit the same
geometric ceiling — 2 rows × L/d_a = 142 cars — because lane-less
spontaneously settles into the same two-row layout that lane-based
imposes. If true, the lane-less "capacity advantage" claim is *false*
in steady state. The advantage from Exp C lives in incident response,
not in raw capacity.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import exp_fundamental_diagram as A


def _initial_lane_based(N, L, W, v_d, seed):
    rng = np.random.default_rng(seed + N)
    usable_lo = A.D_B + 0.5
    usable_hi = W - A.D_B - 0.5
    lane_ys = np.array([usable_lo + (usable_hi - usable_lo) / 4,
                        usable_lo + 3 * (usable_hi - usable_lo) / 4])
    per_lane = [N // 2, N - N // 2]
    q = np.zeros((N, 2))
    lane_y = np.zeros(N)
    idx = 0
    for li, count in enumerate(per_lane):
        if count == 0:
            continue
        xs = (np.arange(count) * L / count + li * (L / count) * 0.5) % L
        q[idx:idx + count, 0] = xs
        q[idx:idx + count, 1] = lane_ys[li] + rng.normal(0, 0.05, count)
        lane_y[idx:idx + count] = lane_ys[li]
        idx += count
    p = np.tile(np.array([v_d, 0.0]), (N, 1))
    return q, p, lane_y


def _initial_lane_less(N, L, W, v_d, seed):
    """Match Exp A's seeding: hex lattice with small jitter if it fits, else
    uniform random."""
    rng = np.random.default_rng(seed + N)
    n_x = max(2, int(np.ceil(L / A.D_A)))
    n_y = max(1, int(np.ceil((W - 2 * A.D_B) / (A.D_A * np.sqrt(3) / 2))))
    candidates = []
    for ix in range(n_x):
        for iy in range(n_y):
            x = ix * A.D_A + (A.D_A / 2 if iy % 2 else 0)
            y = A.D_B + 0.5 + iy * A.D_A * np.sqrt(3) / 2
            if 0 <= x < L and A.D_B + 0.2 < y < W - A.D_B - 0.2:
                candidates.append((x, y))
    if N > len(candidates):
        q = np.zeros((N, 2))
        q[:, 0] = rng.uniform(0, L, N)
        q[:, 1] = rng.uniform(A.D_B + 0.3, W - A.D_B - 0.3, N)
    else:
        idx = rng.choice(len(candidates), N, replace=False)
        q = np.array([candidates[i] for i in idx]) + rng.normal(0, 0.2, (N, 2))
    p = np.tile(np.array([v_d, 0.0]), (N, 1))
    return q, p


def _min_pair_dist(q, L):
    dx = q[None, :, 0] - q[:, None, 0]
    dy = q[None, :, 1] - q[:, None, 1]
    dx = (dx + L / 2) % L - L / 2
    d2 = dx * dx + dy * dy
    n = q.shape[0]
    d2[np.arange(n), np.arange(n)] = np.inf
    return float(np.sqrt(d2.min()))


def run(N, lane_based, L=500.0, W=14.0, v_d=10.0, T=30.0, TS=0.02, seed=42):
    if lane_based:
        q, p, lane_y = _initial_lane_based(N, L, W, v_d, seed)
    else:
        q, p = _initial_lane_less(N, L, W, v_d, seed)
        lane_y = None
    num_steps = int(round(T / TS))
    measure_start = int(num_steps * (2.0 / 3.0))
    vx_samples, std_samples, intra_samples = [], [], []
    for t in range(num_steps):
        u = A.alpha_force_periodic(q, p, L)
        u += A.beta_force_corridor(q, p, W)
        u += A.gamma_force(p, v_d)
        mags = np.linalg.norm(u, axis=1)
        scale = np.where(mags > A.A_MAX, A.A_MAX / np.maximum(mags, 1e-9), 1.0)
        u *= scale[:, None]
        if lane_based:
            u[:, 1] = 0.0
            p[:, 1] = 0.0
            q[:, 1] = lane_y
        p += TS * u
        q += TS * p
        q[:, 0] %= L
        if t >= measure_start:
            vx_samples.append(p[:, 0].mean())
            std_samples.append(p[:, 0].std())
            intra_samples.append(_min_pair_dist(q, L))
    return dict(
        v_mean=float(np.mean(vx_samples)),
        v_std=float(np.mean(std_samples)),
        intra_avg=float(np.mean(intra_samples)),
        intra_worst=float(np.min(intra_samples)),
    )


def main():
    L, W, v_d = 500.0, 14.0, 10.0
    N_vals = [60, 90, 120, 140, 160, 180, 220, 260]
    SAFETY = A.D_A / 2

    print(f'L={L} m, W={W} m, v_d={v_d} m/s, d_a={A.D_A}, safety = d_a/2 = '
          f'{SAFETY:.1f} m\n')
    print(f'{"N":>4}  {"k[v/km]":>8}  '
          f'{"v_ll[km/h]":>10}  {"intra_ll":>10}  '
          f'{"v_lb[km/h]":>10}  {"intra_lb":>10}')
    rows_ll, rows_lb = [], []
    for N in N_vals:
        r_ll = run(N, lane_based=False)
        r_lb = run(N, lane_based=True)
        rows_ll.append((N, r_ll))
        rows_lb.append((N, r_lb))
        k = N / L
        print(f'{N:4d}  {k*1000:8.0f}  '
              f'{r_ll["v_mean"]*3.6:10.3f}  {r_ll["intra_worst"]:10.2f}  '
              f'{r_lb["v_mean"]*3.6:10.3f}  {r_lb["intra_worst"]:10.2f}')

    def capacity(rows):
        N_safe = [N for N, r in rows if r['intra_worst'] >= SAFETY]
        return max(N_safe) if N_safe else 0
    N_ll = capacity(rows_ll)
    N_lb = capacity(rows_lb)
    q_ll = N_ll * v_d / L * 3600
    q_lb = N_lb * v_d / L * 3600
    print('\n=== Capacity comparison (intra-min ≥ d_a/2) ===')
    print(f'  lane_less : N* = {N_ll}, k = {N_ll/L*1000:.0f} veh/km, '
          f'q = {q_ll:.0f} veh/h')
    print(f'  lane_based: N* = {N_lb}, k = {N_lb/L*1000:.0f} veh/km, '
          f'q = {q_lb:.0f} veh/h')
    print(f'  ratio (lane_less / lane_based) = {q_ll / max(q_lb, 1):.2f}×')
    print(f'  strip-hex theoretical limit    = '
          f'{2 * int(L / A.D_A) / L * 1000:.0f} veh/km')

    Ns = np.array([N for N, _ in rows_ll])
    ks = Ns / L * 1000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(ks, [r['intra_worst'] for _, r in rows_ll], '-o', color='C0',
            label='lane_less')
    ax.plot(ks, [r['intra_worst'] for _, r in rows_lb], '-s', color='C3',
            label='lane_based (2 lanes)')
    ax.axhline(A.D_A, color='gray', linestyle=':', label=f'd_a = {A.D_A:.0f}')
    ax.axhline(SAFETY, color='red', linestyle=':',
               label=f'd_a/2 = {SAFETY:.1f}')
    ax.set_xlabel('density k [veh/km]')
    ax.set_ylabel('worst-case intra-min [m]')
    ax.set_title('Lattice integrity vs. density')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1]
    qs_ll = np.array([N * v_d / L * 3600 for N, _ in rows_ll])
    qs_lb = np.array([N * v_d / L * 3600 for N, _ in rows_lb])
    safe_ll = np.array([r['intra_worst'] >= SAFETY for _, r in rows_ll])
    safe_lb = np.array([r['intra_worst'] >= SAFETY for _, r in rows_lb])
    ax.plot(ks[safe_ll], qs_ll[safe_ll], '-o', color='C0', label='lane_less (safe)')
    ax.plot(ks[~safe_ll], qs_ll[~safe_ll], '--o', color='C0', alpha=0.4,
            label='lane_less (unsafe)')
    ax.plot(ks[safe_lb], qs_lb[safe_lb], '-s', color='C3', label='lane_based (safe)')
    ax.plot(ks[~safe_lb], qs_lb[~safe_lb], '--s', color='C3', alpha=0.4,
            label='lane_based (unsafe)')
    if N_ll > 0:
        ax.scatter([N_ll / L * 1000], [q_ll], s=140, color='C0', marker='*',
                   edgecolor='black', zorder=5,
                   label=f'lane_less capacity = {q_ll:.0f}')
    if N_lb > 0:
        ax.scatter([N_lb / L * 1000], [q_lb], s=140, color='C3', marker='*',
                   edgecolor='black', zorder=5,
                   label=f'lane_based capacity = {q_lb:.0f}')
    ax.axhline(8556, color='gray', linestyle='--', alpha=0.5,
               label='HCM 14 m ≈ 8 556')
    ax.set_xlabel('density k [veh/km]')
    ax.set_ylabel('throughput k·v_d [veh/h]')
    ax.set_title('Throughput vs. density (q = k·v_d while safe)')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(
        f'Exp G — honest head-to-head capacity: lane-less vs lane-based '
        f'(same {W:.0f} m corridor, v_d = {v_d:.0f} m/s)',
        y=1.00, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig('exp_capacity_comparison.png', dpi=110, bbox_inches='tight')
    print('\nsaved exp_capacity_comparison.png')


if __name__ == '__main__':
    main()
