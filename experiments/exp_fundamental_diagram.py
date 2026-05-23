"""Experiment A: maximum stable density for the flocking corridor.

Periodic corridor of length L=500 m, width W=14 m, all cars going +x at
v_d=10 m/s. Sweep N across the lattice-break transition. For each N:
seed cars on a hex lattice with jitter, run the α/β/γ system for 30 s,
measure mean v_x AND intra-flock pair-min distance over the last third.

WHY NOT A CLASSICAL q-k-v PLOT (diagnose_fundamental.py finding):
  In a translation-invariant periodic corridor, the α-gradient is
  x-symmetric so mean(α_x) = 0 at steady state. γ = -c_g(v - v_d·x̂) is
  the only x-asymmetric force, so mean(γ_x) = 0 ⇒ mean(v_x) = v_d for
  ALL densities. The algorithm has no congestion regime by construction.
  Past the lattice capacity it fails by overlap (intra-min → 0), not by
  slowing down. The classical fundamental diagram framing is degenerate.

  The honest replacement metric is the maximum density at which the
  α-lattice survives (intra-min stays above a safety threshold). That
  gives a "safe throughput" q_stable = k_stable · v_d to compare with
  HCM ~2200 veh/hr/lane (per-lane, 2 lanes ≈ 4400 veh/hr at W=14 m).

τ doesn't fire in this experiment (all cars are parallel). All cars
share flock_id=1 so α-lattice handles inter-car spacing.

The α-force is vectorized with NumPy for speed (handles N=300 in seconds).
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

# ---- constants ---------------------------------------------------------
EPS = 0.1
D_A = 7.0
R_A = 1.2 * D_A
H_A = 0.2
C1_A = 5.0
C2_A = 2.0 * np.sqrt(5)
D_B = 3.0
H_B = 0.2
C1_B = 200.0
C2_B = 2.0 * np.sqrt(200)
C_G = 1.5
A_MAX = 9.0
A_PHI = 5.0  # phi(z) shape parameter (a == b => phi = a * z / sqrt(1+z²))


def _sigma_norm_scalar(z, e):
    return (np.sqrt(1 + e * z * z) - 1) / e


SIGMA_R_A = _sigma_norm_scalar(R_A, EPS)
SIGMA_D_A = _sigma_norm_scalar(D_A, EPS)
SIGMA_D_B = _sigma_norm_scalar(D_B, EPS)


def alpha_force_periodic(q, p, L):
    """Vectorized α-force over all pairs with periodic boundary in x.
    Returns (N, 2) array of accelerations.
    """
    N = q.shape[0]
    # Pairwise displacement: dij = q[j] - q[i], shape (N, N, 2)
    dx = q[None, :, 0] - q[:, None, 0]
    dy = q[None, :, 1] - q[:, None, 1]
    # min-image x-wrap
    dx = (dx + L / 2) % L - L / 2
    d2 = dx * dx + dy * dy
    # σ-norm of displacement
    z = (np.sqrt(1 + EPS * d2) - 1) / EPS
    # mask: same-cell exclusion + within interaction range
    diag = np.eye(N, dtype=bool)
    active = (z <= SIGMA_R_A) & (~diag)
    # ρ_h_a(z / σ(r_a)) — bump function
    arg = z / SIGMA_R_A
    rho_v = np.where(arg < H_A, 1.0,
                     np.where(arg <= 1.0,
                              0.5 * (1 + np.cos(np.pi * (arg - H_A) / (1 - H_A))),
                              0.0))
    # φ(z - σ(d_a)) with a == b => simplified
    zd = z - SIGMA_D_A
    phi_v = A_PHI * zd / np.sqrt(1 + zd * zd)
    phi_alpha = rho_v * phi_v
    # n_ij = (q_j - q_i) / sqrt(1 + ε|q_j - q_i|²)
    n_denom = np.sqrt(1 + EPS * d2)
    nx = dx / n_denom
    ny = dy / n_denom
    grad_x = (phi_alpha * nx * active).sum(axis=1)
    grad_y = (phi_alpha * ny * active).sum(axis=1)
    # a_ij = ρ_h_a(σ_ε(q_j - q_i) / σ_ε(r_a)) — reuse rho_v
    a_ij = rho_v
    dvx = p[None, :, 0] - p[:, None, 0]
    dvy = p[None, :, 1] - p[:, None, 1]
    cons_x = (a_ij * dvx * active).sum(axis=1)
    cons_y = (a_ij * dvy * active).sum(axis=1)
    return C1_A * np.stack([grad_x, grad_y], axis=1) + \
           C2_A * np.stack([cons_x, cons_y], axis=1)


def beta_force_corridor(q, p, W):
    """β walls at y=0 and y=W. Returns (N, 2) array."""
    N = q.shape[0]
    ui = np.zeros((N, 2))
    for wall_y, inward in [(0.0, +1.0), (W, -1.0)]:
        abs_dist = np.abs(q[:, 1] - wall_y)
        z = _sigma_norm_scalar(abs_dist, EPS)
        active = z <= SIGMA_D_B
        # φ_β
        zd = z - SIGMA_D_B
        s1 = zd / np.sqrt(1 + zd * zd)
        # ρ for bump
        arg = z / SIGMA_D_B
        rho_v = np.where(arg < H_B, 1.0,
                         np.where(arg <= 1.0,
                                  0.5 * (1 + np.cos(np.pi * (arg - H_B) / (1 - H_B))),
                                  0.0))
        phi_beta_v = rho_v * (s1 - 1.0)
        mag = -phi_beta_v  # magnitude (positive)
        ui[:, 1] += np.where(active, C1_B * mag * inward, 0.0)
        bik = rho_v
        ui[:, 1] += np.where(active, C2_B * bik * (0.0 - p[:, 1]), 0.0)
    return ui


def gamma_force(p, v_d):
    return -C_G * (p - np.array([v_d, 0.0]))


def _min_pair_distance_periodic(q, L):
    dx = q[None, :, 0] - q[:, None, 0]
    dy = q[None, :, 1] - q[:, None, 1]
    dx = (dx + L / 2) % L - L / 2
    d2 = dx * dx + dy * dy
    N = q.shape[0]
    d2[np.arange(N), np.arange(N)] = np.inf
    return float(np.sqrt(d2.min()))


def run_periodic_corridor(N, L, W, v_d, T=30.0, TS=0.02, seed=42):
    """Returns dict with mean v_x, std v_x, time-avg intra-min, worst-frame intra-min."""
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
    vx_samples, std_samples, intra_samples = [], [], []
    for t in range(num_steps):
        u = alpha_force_periodic(q, p, L)
        u += beta_force_corridor(q, p, W)
        u += gamma_force(p, v_d)
        mags = np.linalg.norm(u, axis=1)
        scale = np.where(mags > A_MAX, A_MAX / np.maximum(mags, 1e-9), 1.0)
        u *= scale[:, None]
        p += TS * u
        q += TS * p
        q[:, 0] %= L
        if t >= measure_start:
            vx_samples.append(p[:, 0].mean())
            std_samples.append(p[:, 0].std())
            intra_samples.append(_min_pair_distance_periodic(q, L))
    return {
        'v_mean': float(np.mean(vx_samples)),
        'v_std': float(np.mean(std_samples)),
        'intra_avg': float(np.mean(intra_samples)),
        'intra_worst': float(np.min(intra_samples)),
    }


def main():
    L, W = 500.0, 14.0
    v_d = 10.0
    # Fine sweep across the lattice-break transition (seen 240 → 500 veh/km in
    # diagnose_fundamental.py). Sparse high-N points just to show the
    # post-collapse plateau.
    N_vals = [10, 30, 60, 90, 120, 140, 160, 180, 200, 220, 240,
              260, 280, 300, 350, 500, 700]
    SAFETY = D_A / 2  # 3.5 m — half of d_a; loosely "car-width" safety threshold

    print(f'L={L} m, W={W} m, v_d={v_d} m/s, d_a={D_A} m')
    print(f'Safety threshold for k_stable: intra-min ≥ {SAFETY:.1f} m')
    print(f'{"N":>4}  {"k[v/km]":>8}  {"v[km/h]":>8}  {"σv[km/h]":>8}  '
          f'{"intra_avg":>9}  {"intra_worst":>11}  {"q[v/h]":>8}')
    rows = []
    for N in N_vals:
        r = run_periodic_corridor(N, L, W, v_d, T=30.0)
        k = N / L
        q = k * r['v_mean']
        rows.append((N, k, r['v_mean'], r['v_std'], r['intra_avg'],
                     r['intra_worst'], q))
        print(f'{N:4d}  {k*1000:8.1f}  {r["v_mean"]*3.6:8.3f}  '
              f'{r["v_std"]*3.6:8.3f}  {r["intra_avg"]:9.2f}  '
              f'{r["intra_worst"]:11.2f}  {q*3600:8.0f}')

    Ns = np.array([r[0] for r in rows])
    ks = np.array([r[1] for r in rows])
    vs = np.array([r[2] for r in rows])
    sigs = np.array([r[3] for r in rows])
    intra_avg = np.array([r[4] for r in rows])
    intra_worst = np.array([r[5] for r in rows])
    qs = np.array([r[6] for r in rows])

    # k_stable = largest k where worst-frame intra-min ≥ SAFETY.
    safe_mask = intra_worst >= SAFETY
    if not safe_mask.any():
        k_stable, q_stable, N_stable = 0.0, 0.0, 0
    else:
        idx = np.where(safe_mask)[0][-1]
        k_stable = ks[idx]
        q_stable = ks[idx] * v_d  # mean v_x is pinned at v_d in the safe regime
        N_stable = Ns[idx]
    # also report the looser threshold "lattice still d_a-like" (intra ≥ 0.9*d_a)
    lattice_mask = intra_worst >= 0.9 * D_A
    if lattice_mask.any():
        idx2 = np.where(lattice_mask)[0][-1]
        k_lattice, q_lattice = ks[idx2], ks[idx2] * v_d
    else:
        k_lattice, q_lattice = 0.0, 0.0

    # Strip-hex ceiling: with row spacing d_a·√3/2 and width W_usable, count
    # how many rows fit, then row length = L / d_a (with the cylindrical x).
    W_usable = W - 2 * D_B
    n_rows = max(1, int(np.floor(W_usable / (D_A * np.sqrt(3) / 2))) + 1)
    cars_per_row = L / D_A
    k_strip_theory = n_rows * cars_per_row / L * 1000  # veh/km

    # HCM reference scaled to road width (2200 veh/hr per 3.6 m lane).
    n_lanes_equivalent = W / 3.6
    hcm_equiv = 2200 * n_lanes_equivalent

    print('\n=== Findings ===')
    print(f'  Mean v_x is pinned at v_d = {v_d*3.6:.1f} km/h for ALL densities')
    print(f'  (see diagnose_fundamental.py for why — symmetric α + constant-v_d γ).')
    print(f'  Algorithm has no congestion regime: past lattice capacity it')
    print(f'  fails by overlap (intra-min → 0), not by slowing.\n')
    print(f'  Safe throughput   q_stable   = {q_stable*3600:.0f} veh/h  '
          f'@ k_stable  = {k_stable*1000:.0f} veh/km (N={N_stable})')
    print(f'  Lattice-intact    q_lattice  = {q_lattice*3600:.0f} veh/h  '
          f'@ k_lattice = {k_lattice*1000:.0f} veh/km')
    print(f'  Strip-hex theory             = {k_strip_theory:.0f} veh/km  '
          f'({n_rows} rows × {cars_per_row:.0f} cars on W_usable={W_usable:.0f} m)')
    print(f'  HCM equivalent ({W:.0f} m ≈ {n_lanes_equivalent:.1f} lanes) '
          f'= {hcm_equiv:.0f} veh/h')
    print(f'  q_stable / HCM equivalent    = {q_stable*3600/hcm_equiv:.2f}×')

    # ---- plots ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(ks * 1000, vs * 3.6, '-o', color='C0', label='mean v_x')
    ax.fill_between(ks * 1000, (vs - sigs) * 3.6, (vs + sigs) * 3.6,
                    alpha=0.2, color='C0', label='±1σ')
    ax.axhline(v_d * 3.6, color='gray', linestyle=':',
               label=f'v_d = {v_d*3.6:.0f} km/h')
    ax.set_xlabel('density k [veh/km]')
    ax.set_ylabel('forward speed v_x [km/h]')
    ax.set_title('Speed is pinned at v_d')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ks * 1000, intra_avg, '-o', color='C1', label='intra-min (time-avg)')
    ax.plot(ks * 1000, intra_worst, '-s', color='C3', label='intra-min (worst frame)')
    ax.axhline(D_A, color='gray', linestyle=':', label=f'd_a = {D_A:.0f}')
    ax.axhline(SAFETY, color='red', linestyle=':',
               label=f'safety = d_a/2 = {SAFETY:.1f}')
    ax.axvline(k_stable * 1000, color='black', linestyle='--', alpha=0.5,
               label=f'k_stable = {k_stable*1000:.0f}')
    ax.set_xlabel('density k [veh/km]')
    ax.set_ylabel('pair-min distance [m]')
    ax.set_title('Lattice survives until k_stable, then collapses')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    # q = k·v plotted, with the unsafe regime drawn in red and the safe regime
    # green. q_stable highlighted.
    safe = intra_worst >= SAFETY
    ax.plot(ks[safe] * 1000, qs[safe] * 3600, '-o', color='C2', label='safe')
    ax.plot(ks[~safe] * 1000, qs[~safe] * 3600, '-o', color='C3',
            label='unsafe (overlapping)')
    ax.scatter([k_stable * 1000], [q_stable * 3600], s=120, color='black',
               zorder=5, marker='*',
               label=f'q_stable = {q_stable*3600:.0f} veh/h')
    ax.axhline(hcm_equiv, color='gray', linestyle=':',
               label=f'HCM {n_lanes_equivalent:.1f}-lane ≈ {hcm_equiv:.0f} veh/h')
    ax.set_xlabel('density k [veh/km]')
    ax.set_ylabel('flow q = k·v_x [veh/h]')
    ax.set_title('Throughput — safe vs. unsafe regime')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Exp A — max stable density for flocking corridor '
        f'(L={L:.0f} m, W={W:.0f} m, v_d={v_d:.0f} m/s, d_a={D_A:.0f} m)',
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig('exp_fundamental_diagram.png', dpi=110, bbox_inches='tight')
    print('\nsaved exp_fundamental_diagram.png')


if __name__ == '__main__':
    main()
