"""Experiment D: macroscopic fundamental diagram on a 4-arm intersection.

Open square arena, no β walls. Four arms (E, W, N, S) inject cars heading
toward the centre at a rate λ veh/s/arm. Cars exit when they cross the
opposite boundary. Sweep λ; for each λ measure the realised throughput,
the drop rate (blocked injections), and the worst-case safety margin.

Algorithm: V2 mode-aware for intersections (per investigations_queue.md):
  - α + γ standard.
  - τ uses true rotation matrix R(+90°), velocity-only (c1_t=0, c2_t=0.15),
    d_c=70. Predictive suppression OFF (intersection mode).
  - No β (walls placed at infinity).

Each arm has its own flock id and its own desired velocity (toward the
opposite arm), so γ pulls each car in its arm's direction. Cars from
different arms see each other through τ.

Comparison reference: signalised intersection with saturation flow rate
s = 1800 veh/h/lane and 4-phase signal (1 phase per arm, 25% green each)
gives per-arm capacity ≈ 450 veh/h ⇒ ~1800 veh/h total. The flocking
intersection allows ALL arms to flow simultaneously, so it should beat
that handily — the question is by how much, and at what safety cost.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

# ---- algorithm constants (matched to exp_fundamental_diagram.py) -------
EPS = 0.1
D_A = 7.0
R_A = 1.2 * D_A
H_A = 0.2
C1_A = 5.0
C2_A = 2.0 * np.sqrt(5.0)
A_PHI = 5.0
C_G = 1.5
A_MAX = 9.0
# V2 intersection tau parameters
D_C = 70.0
C1_T = 0.0
C2_T = 0.15


def _sigma_norm_scalar(z, e):
    return (np.sqrt(1 + e * z * z) - 1) / e


SIGMA_R_A = _sigma_norm_scalar(R_A, EPS)
SIGMA_D_A = _sigma_norm_scalar(D_A, EPS)

# R(+90°): rotates a vector 90° counter-clockwise.  R @ [vx, vy] = [-vy, vx].
# Substituted for McKenzie's reflection J — see investigations_queue.md #7.

# ---- arena ------------------------------------------------------------
L_HALF = 60.0
V_D = 10.0
# spawn_pos, vel, flock_id, exit-test-fn
ARMS = [
    ('E', np.array([+L_HALF, 0.0]), np.array([-V_D, 0.0]), 1),
    ('W', np.array([-L_HALF, 0.0]), np.array([+V_D, 0.0]), 2),
    ('N', np.array([0.0, +L_HALF]), np.array([0.0, -V_D]), 3),
    ('S', np.array([0.0, -L_HALF]), np.array([0.0, +V_D]), 4),
]


def _alpha_force(q, p, fid):
    N = q.shape[0]
    if N < 2:
        return np.zeros((N, 2))
    dx = q[None, :, 0] - q[:, None, 0]
    dy = q[None, :, 1] - q[:, None, 1]
    d2 = dx * dx + dy * dy
    z = (np.sqrt(1 + EPS * d2) - 1) / EPS
    same = fid[:, None] == fid[None, :]
    diag = np.eye(N, dtype=bool)
    active = (z <= SIGMA_R_A) & ~diag & same
    arg = z / SIGMA_R_A
    rho = np.where(arg < H_A, 1.0,
                   np.where(arg <= 1.0,
                            0.5 * (1 + np.cos(np.pi * (arg - H_A) / (1 - H_A))),
                            0.0))
    zd = z - SIGMA_D_A
    phi_v = A_PHI * zd / np.sqrt(1 + zd * zd)
    phi_alpha = rho * phi_v
    n_denom = np.sqrt(1 + EPS * d2)
    nx = dx / n_denom
    ny = dy / n_denom
    grad_x = (phi_alpha * nx * active).sum(axis=1)
    grad_y = (phi_alpha * ny * active).sum(axis=1)
    a_ij = rho
    dvx = p[None, :, 0] - p[:, None, 0]
    dvy = p[None, :, 1] - p[:, None, 1]
    cons_x = (a_ij * dvx * active).sum(axis=1)
    cons_y = (a_ij * dvy * active).sum(axis=1)
    return (C1_A * np.stack([grad_x, grad_y], axis=1)
            + C2_A * np.stack([cons_x, cons_y], axis=1))


def _tau_force(q, p, fid):
    """R(+90°), velocity-only τ, gated by d_c and opposing headings."""
    N = q.shape[0]
    if N < 2:
        return np.zeros((N, 2))
    diff_qx = q[:, None, 0] - q[None, :, 0]
    diff_qy = q[:, None, 1] - q[None, :, 1]
    dist = np.sqrt(diff_qx * diff_qx + diff_qy * diff_qy)
    diff_px = p[:, None, 0] - p[None, :, 0]
    diff_py = p[:, None, 1] - p[None, :, 1]
    same = fid[:, None] == fid[None, :]
    diag = np.eye(N, dtype=bool)
    within = dist <= D_C
    pmag = np.linalg.norm(p, axis=1)
    pmag_safe = pmag + 1e-9
    cos_pp = (p[:, None, 0] * p[None, :, 0] + p[:, None, 1] * p[None, :, 1]) \
        / (pmag_safe[:, None] * pmag_safe[None, :])
    opposing = cos_pp <= 0
    active = ~same & ~diag & within & opposing
    # u_i contribution from r:  -c2_t · R · (p_i - p_r) where R = R_PLUS_90.
    # R · [a, b] = [-b, a]. With diff_p_x = p_i.x - p_r.x:
    #   force_x = -c2_t · (-diff_p_y) =  c2_t · diff_p_y
    #   force_y = -c2_t ·   diff_p_x  = -c2_t · diff_p_x
    fx_term = +diff_py * active
    fy_term = -diff_px * active
    ux = C2_T * fx_term.sum(axis=1)
    uy = C2_T * fy_term.sum(axis=1)
    return np.stack([ux, uy], axis=1)


def _gamma_force(p, fid, arm_p_d):
    N = p.shape[0]
    if N == 0:
        return np.zeros((0, 2))
    u = np.zeros((N, 2))
    for f in (1, 2, 3, 4):
        mask = fid == f
        if mask.any():
            u[mask] = -C_G * (p[mask] - arm_p_d[f][None, :])
    return u


def _saturate(u):
    mag = np.linalg.norm(u, axis=1)
    scale = np.where(mag > A_MAX, A_MAX / np.maximum(mag, 1e-9), 1.0)
    return u * scale[:, None]


def run_intersection_mfd(lambda_per_arm, T=60.0, DT=0.02, seed=42,
                         spawn_clear=0.9 * D_A):
    rng = np.random.default_rng(seed)
    arm_p_d = {fid: vel for _, _, vel, fid in ARMS}
    period = 1.0 / lambda_per_arm

    q = np.empty((0, 2))
    p = np.empty((0, 2))
    fid = np.empty(0, dtype=int)
    t_enter = np.empty(0)

    next_inject = np.zeros(4)
    n_completed_arm = np.zeros(4, dtype=int)
    n_dropped_arm = np.zeros(4, dtype=int)
    n_injected_arm = np.zeros(4, dtype=int)

    min_pair_dist = np.inf
    travel_times = []
    pop_history = []

    num_steps = int(T / DT)
    for t_idx in range(num_steps):
        t = t_idx * DT
        # ---- injection at each arm -------------------------------------
        for arm_idx, (_, spawn, vel, f_id) in enumerate(ARMS):
            if t >= next_inject[arm_idx]:
                if q.shape[0] == 0:
                    free = True
                else:
                    free = np.all(np.linalg.norm(q - spawn[None], axis=1)
                                  > spawn_clear)
                if free:
                    jitter_perp = rng.normal(0, 0.2)
                    # jitter perpendicular to motion
                    if vel[0] != 0:
                        spawn_pt = spawn + np.array([0.0, jitter_perp])
                    else:
                        spawn_pt = spawn + np.array([jitter_perp, 0.0])
                    q = np.vstack([q, spawn_pt])
                    p = np.vstack([p, vel])
                    fid = np.append(fid, f_id)
                    t_enter = np.append(t_enter, t)
                    n_injected_arm[arm_idx] += 1
                else:
                    n_dropped_arm[arm_idx] += 1
                next_inject[arm_idx] += period

        if q.shape[0] > 0:
            u = _alpha_force(q, p, fid)
            u = u + _tau_force(q, p, fid)
            u = u + _gamma_force(p, fid, arm_p_d)
            u = _saturate(u)
            p = p + DT * u
            q = q + DT * p

            if q.shape[0] >= 2:
                dx = q[None, :, 0] - q[:, None, 0]
                dy = q[None, :, 1] - q[:, None, 1]
                d2 = dx * dx + dy * dy
                np.fill_diagonal(d2, np.inf)
                m = np.sqrt(d2.min())
                if m < min_pair_dist:
                    min_pair_dist = m

            # ---- exits -------------------------------------------------
            exit_E = (fid == 1) & (q[:, 0] < -L_HALF)
            exit_W = (fid == 2) & (q[:, 0] > L_HALF)
            exit_N = (fid == 3) & (q[:, 1] < -L_HALF)
            exit_S = (fid == 4) & (q[:, 1] > L_HALF)
            for arm_idx, exit_mask in enumerate([exit_E, exit_W, exit_N, exit_S]):
                if exit_mask.any():
                    n_completed_arm[arm_idx] += int(exit_mask.sum())
                    travel_times.extend((t - t_enter[exit_mask]).tolist())
            exit_any = exit_E | exit_W | exit_N | exit_S
            if exit_any.any():
                keep = ~exit_any
                q = q[keep]; p = p[keep]; fid = fid[keep]
                t_enter = t_enter[keep]

        if t_idx % 10 == 0:
            pop_history.append((t, q.shape[0]))

    return {
        'lambda_per_arm': lambda_per_arm,
        'n_injected': int(n_injected_arm.sum()),
        'n_completed': int(n_completed_arm.sum()),
        'n_dropped': int(n_dropped_arm.sum()),
        'n_active_end': q.shape[0],
        'throughput_total': n_completed_arm.sum() / T,     # veh/s
        'throughput_per_arm': n_completed_arm.sum() / (4 * T),
        'min_pair_dist': float(min_pair_dist) if np.isfinite(min_pair_dist)
                         else np.nan,
        'mean_travel_time': float(np.mean(travel_times)) if travel_times else np.nan,
        'p95_travel_time': float(np.percentile(travel_times, 95)) if travel_times else np.nan,
        'pop_history': np.array(pop_history),
    }


def main():
    lambdas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    print(f'arena = 2·L_HALF = {2*L_HALF} m, v_d = {V_D} m/s, d_a = {D_A} m')
    print(f'physical single-lane spawn cap per arm = v_d/d_a = '
          f'{V_D/D_A:.2f} veh/s = {V_D/D_A*3600:.0f} veh/h\n')

    print(f'{"λ_in":>5}  {"λ_total":>8}  {"q_out":>7}  {"drop%":>6}  '
          f'{"intra":>6}  {"travel":>8}  {"p95 travel":>10}')
    rows = []
    for lam in lambdas:
        r = run_intersection_mfd(lam, T=60.0)
        drop_pct = 100.0 * r['n_dropped'] / max(1, r['n_dropped'] + r['n_injected'])
        rows.append(r)
        print(f'{lam:5.2f}  {4*lam:8.2f}  {r["throughput_total"]:7.2f}  '
              f'{drop_pct:5.1f}%  {r["min_pair_dist"]:6.2f}  '
              f'{r["mean_travel_time"]:8.2f}  {r["p95_travel_time"]:10.2f}')

    # extract capacity: largest λ where drop pct < 5% AND intra-min > d_a/2
    capacity_idx = None
    for i, (lam, r) in enumerate(zip(lambdas, rows)):
        drop_pct = 100.0 * r['n_dropped'] / max(1, r['n_dropped'] + r['n_injected'])
        safe = r['min_pair_dist'] > D_A / 2
        if drop_pct < 5.0 and safe:
            capacity_idx = i
    if capacity_idx is None:
        lam_cap, q_cap = 0.0, 0.0
    else:
        lam_cap = lambdas[capacity_idx]
        q_cap = rows[capacity_idx]['throughput_total']

    # signalised reference: 1800 veh/h/lane × 4 arms × (1/4 phase) = 1800 veh/h total
    signal_total_vph = 1800.0  # veh/h
    signal_total_vps = signal_total_vph / 3600.0  # ≈ 0.5

    print('\n=== Findings ===')
    print(f'  Capacity (drops<5%, intra>{D_A/2:.1f} m): '
          f'λ ≤ {lam_cap:.2f} veh/s/arm  ⇒  '
          f'q = {q_cap:.2f} veh/s = {q_cap*3600:.0f} veh/h total')
    print(f'  Signalised reference (4-phase, s=1800/lane): '
          f'~{signal_total_vph:.0f} veh/h')
    if q_cap > 0:
        print(f'  Flocking / signalised = {q_cap*3600/signal_total_vph:.2f}×')
    print(f'  Single-lane physical cap (4 arms × v_d/d_a): '
          f'~{4*V_D/D_A*3600:.0f} veh/h')

    # ---- plot -----------------------------------------------------------
    lams = np.array(lambdas)
    qs = np.array([r['throughput_total'] for r in rows])
    drops = np.array([100 * r['n_dropped'] / max(1, r['n_dropped'] + r['n_injected'])
                      for r in rows])
    intras = np.array([r['min_pair_dist'] for r in rows])
    travels = np.array([r['mean_travel_time'] for r in rows])
    p95_travels = np.array([r['p95_travel_time'] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(4 * lams * 3600, qs * 3600, '-o', color='C2', label='realised q_out')
    ax.plot([0, 4 * lams.max() * 3600], [0, 4 * lams.max() * 3600], ':',
            color='gray', label='y = x (no drops)')
    if q_cap > 0:
        ax.scatter([4 * lam_cap * 3600], [q_cap * 3600], s=120, color='black',
                   zorder=5, marker='*',
                   label=f'capacity = {q_cap*3600:.0f} veh/h')
    ax.axhline(signal_total_vph, color='gray', linestyle='--',
               label=f'signalised ≈ {signal_total_vph:.0f} veh/h')
    ax.set_xlabel('total demand 4λ [veh/h]')
    ax.set_ylabel('realised q_out [veh/h]')
    ax.set_title('Intersection MFD: demand vs. realised throughput')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(lams, drops, '-o', color='C3', label='drop %')
    ax.set_xlabel('λ per arm [veh/s]')
    ax.set_ylabel('drop % (blocked injections)')
    ax.axhline(5.0, color='gray', linestyle=':', label='5% drop threshold')
    ax.set_title('Injection drops: queue at arm fills up')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(lams, intras, '-o', color='C4', label='min pair dist over run')
    ax.axhline(D_A, color='gray', linestyle=':', label=f'd_a = {D_A:.0f}')
    ax.axhline(D_A / 2, color='red', linestyle=':',
               label=f'd_a/2 = {D_A/2:.1f}  (safety)')
    ax.set_xlabel('λ per arm [veh/s]')
    ax.set_ylabel('worst-case intra-min [m]')
    ax.set_title('Safety: minimum pair distance over the run')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1, 1]
    free_flow_t = 2 * L_HALF / V_D
    ax.plot(lams, travels, '-o', color='C5', label='mean travel time')
    ax.plot(lams, p95_travels, '-s', color='C5', linestyle='--', alpha=0.7,
            label='p95 travel time')
    ax.axhline(free_flow_t, color='gray', linestyle=':',
               label=f'free-flow = {free_flow_t:.1f} s')
    ax.set_xlabel('λ per arm [veh/s]')
    ax.set_ylabel('travel time [s]')
    ax.set_title('Delay: time from arm to opposite arm')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle(
        f'Exp D — intersection MFD  (open arena {2*L_HALF:.0f} m × {2*L_HALF:.0f} m, '
        f'V2 τ with R(+90°), c2_t={C2_T}, d_c={D_C:.0f})',
        y=1.00, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig('exp_intersection_mfd.png', dpi=110, bbox_inches='tight')
    print('\nsaved exp_intersection_mfd.png')


if __name__ == '__main__':
    main()
