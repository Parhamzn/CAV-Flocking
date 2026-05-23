"""Experiment C: lane-less vs lane-based capacity & perturbation recovery.

Setup: same periodic corridor (L=500 m, W=14 m, v_d=10 m/s). Two lanes
centred at y = (D_B + W/2)/2 and y = (W + D_B + W/2)/2 — equivalently
y_low and y_high inside the β-free strip. Initial positions place N/2
cars per lane on an x-grid of pitch d_a; y is the lane centre (with a
tiny jitter so α doesn't see a degenerate co-linear configuration).

Two conditions sharing seed + initial positions + initial velocity:
  lane_based : y_velocity zeroed each step, y_position reset to the
               initial lane centre — every car is pinned to its lane.
               x-dynamics still get α + γ.
  lane_less  : full α + β + γ on a 2-D plane. Cars may change "lane"
               freely if α prefers it.

Phases of each run (T = 30 s):
   0 –  8 s  settle
   8 – 10 s  perturb: hold car 0 at v_x = v_brake (2 m/s) by overriding
             its forces and velocity each step. Reproduces a real-world
             slow-down of one driver.
  10 – 30 s  release. Measure recovery.

Per-frame quantities recorded:
  mean_vx, min_vx, wake_count (#cars with v_x < 0.9 v_d), intra_min.

Recovery metrics derived from the post-release window:
  t_full_recovery : first t > 10 s with wake_count == 0 for ≥ 1 s
  t_to_9d         : first t > 10 s with mean_vx ≥ 0.99·v_d
  max_wake        : peak wake_count during/after perturbation
  intra_worst     : worst intra-min during the whole run

Predicted advantage for lane-less: it can route around the slow car
laterally, so wake stays small and recovery is fast. Lane-based has no
y-freedom; cars behind the brake pile up until the brake releases and
even then the α-x repulsion has to push everyone forward in single file.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from exp_fundamental_diagram import (
    alpha_force_periodic, beta_force_corridor, gamma_force,
    D_A, D_B, A_MAX,
)


def _min_pair_distance_periodic(q, L):
    dx = q[None, :, 0] - q[:, None, 0]
    dy = q[None, :, 1] - q[:, None, 1]
    dx = (dx + L / 2) % L - L / 2
    d2 = dx * dx + dy * dy
    N = q.shape[0]
    d2[np.arange(N), np.arange(N)] = np.inf
    return float(np.sqrt(d2.min()))


def _initial_state(N, L, W, v_d, n_lanes=2, seed=42):
    """Place N cars across n_lanes equal lanes on an x-grid of pitch d_a.
    Returns q (N,2), p (N,2), lane_y (N,) — the assigned lane y for each car.
    """
    rng = np.random.default_rng(seed + N)
    usable_lo, usable_hi = D_B + 0.5, W - D_B - 0.5
    # n_lanes evenly spaced lane centres inside the usable strip
    lane_ys = np.linspace(usable_lo + (usable_hi - usable_lo) / (2 * n_lanes),
                          usable_hi - (usable_hi - usable_lo) / (2 * n_lanes),
                          n_lanes)
    per_lane = [N // n_lanes] * n_lanes
    for i in range(N - sum(per_lane)):
        per_lane[i] += 1
    q = np.zeros((N, 2))
    lane_y = np.zeros(N)
    idx = 0
    for li, count in enumerate(per_lane):
        if count == 0:
            continue
        x_pitch = L / count
        for j in range(count):
            q[idx, 0] = (j * x_pitch + (li * x_pitch / n_lanes)) % L
            q[idx, 1] = lane_ys[li] + rng.normal(0, 0.05)
            lane_y[idx] = lane_ys[li]
            idx += 1
    p = np.tile(np.array([v_d, 0.0]), (N, 1))
    return q, p, lane_y


def run_corridor_perturb(N, L, W, v_d, lane_based,
                         T=30.0, TS=0.02, seed=42,
                         t_brake_start=8.0, t_brake_end=10.0,
                         v_brake=2.0, brake_idx=0):
    q, p, lane_y = _initial_state(N, L, W, v_d, n_lanes=2, seed=seed)
    num_steps = int(round(T / TS))
    log = {'t': [], 'mean_vx': [], 'min_vx': [], 'wake': [], 'intra_min': []}
    threshold = 0.9 * v_d
    for t in range(num_steps):
        time = t * TS
        u = alpha_force_periodic(q, p, L)
        u += beta_force_corridor(q, p, W)
        u += gamma_force(p, v_d)
        mags = np.linalg.norm(u, axis=1)
        scale = np.where(mags > A_MAX, A_MAX / np.maximum(mags, 1e-9), 1.0)
        u *= scale[:, None]
        # perturbation: force brake on car `brake_idx` between t_brake_start..end
        if t_brake_start <= time < t_brake_end:
            p[brake_idx, 0] = v_brake
            p[brake_idx, 1] = 0.0
            u[brake_idx] = 0.0
        if lane_based:
            u[:, 1] = 0.0
            p[:, 1] = 0.0
            q[:, 1] = lane_y  # hard-reset to lane centre
        p += TS * u
        q += TS * p
        q[:, 0] %= L
        log['t'].append(time)
        log['mean_vx'].append(p[:, 0].mean())
        log['min_vx'].append(p[:, 0].min())
        log['wake'].append(int((p[:, 0] < threshold).sum()))
        log['intra_min'].append(_min_pair_distance_periodic(q, L))
    return {k: np.array(v) for k, v in log.items()}


def derive_metrics(log, v_d, t_release=10.0, settle_s=1.0):
    t = log['t']
    post = t >= t_release
    # peak wake during/after perturbation (skip settle)
    perturb_window = (t >= 7.0)
    max_wake = int(log['wake'][perturb_window].max())
    # full recovery: first sustained interval of wake==0 lasting `settle_s`
    wake_post = log['wake'][post]
    t_post = t[post]
    TS = t[1] - t[0]
    window = int(round(settle_s / TS))
    t_full_recovery = np.nan
    for i in range(len(wake_post) - window):
        if (wake_post[i:i + window] == 0).all():
            t_full_recovery = t_post[i]
            break
    # mean_vx recovery to 99% v_d
    mean_post = log['mean_vx'][post]
    rec_idx = np.where(mean_post >= 0.99 * v_d)[0]
    t_to_99 = t_post[rec_idx[0]] if len(rec_idx) else np.nan
    intra_worst = float(log['intra_min'][perturb_window].min())
    return {
        'max_wake': max_wake,
        't_full_recovery': float(t_full_recovery),
        't_to_99_v_d': float(t_to_99),
        'intra_worst': intra_worst,
        'recovery_duration': float(t_full_recovery - t_release)
            if not np.isnan(t_full_recovery) else np.nan,
    }


def main():
    L, W, v_d = 500.0, 14.0, 10.0
    N_vals = [60, 90, 120, 140]

    print(f'L={L}  W={W}  v_d={v_d}   2 lanes at y≈{D_B+(W-2*D_B)/4:.1f}, '
          f'{W-D_B-(W-2*D_B)/4:.1f}')
    print(f'Perturbation: brake car 0 to v_x={2.0} m/s during t=[8,10] s; '
          f'release at t=10 s; total run 30 s.\n')
    print(f'{"N":>4} {"cond":>11}  {"max_wake":>8}  '
          f'{"t_recov":>8}  {"Δt_recov":>10}  {"t_to_99":>8}  '
          f'{"intra_worst":>11}')
    all_logs = {}
    summary = {'lane_based': [], 'lane_less': []}
    for N in N_vals:
        for cond_name, lane_based in [('lane_based', True), ('lane_less', False)]:
            log = run_corridor_perturb(N, L, W, v_d, lane_based=lane_based)
            metrics = derive_metrics(log, v_d)
            all_logs[(N, cond_name)] = log
            summary[cond_name].append((N, metrics))
            recov = f'{metrics["t_full_recovery"]:8.2f}' \
                if not np.isnan(metrics['t_full_recovery']) else '   ----'
            dur = f'{metrics["recovery_duration"]:10.2f}' \
                if not np.isnan(metrics['recovery_duration']) else '      ----'
            t99 = f'{metrics["t_to_99_v_d"]:8.2f}' \
                if not np.isnan(metrics['t_to_99_v_d']) else '   ----'
            print(f'{N:4d} {cond_name:>11}  {metrics["max_wake"]:8d}  '
                  f'{recov}  {dur}  {t99}  '
                  f'{metrics["intra_worst"]:11.2f}')

    # ---- plots ---------------------------------------------------------
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 4)

    # row 0–1: per-N timeseries of wake count for both conditions
    for j, N in enumerate(N_vals):
        ax = fig.add_subplot(gs[0, j])
        for cond_name, color in [('lane_based', 'C3'), ('lane_less', 'C0')]:
            log = all_logs[(N, cond_name)]
            ax.plot(log['t'], log['wake'], color=color, label=cond_name)
        ax.axvspan(8, 10, color='gray', alpha=0.15, label='brake')
        ax.set_title(f'N={N}: wake count (# cars below 0.9 v_d)')
        ax.set_xlabel('time [s]'); ax.set_ylabel('# cars')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[1, j])
        for cond_name, color in [('lane_based', 'C3'), ('lane_less', 'C0')]:
            log = all_logs[(N, cond_name)]
            ax2.plot(log['t'], log['mean_vx'], color=color, label=cond_name)
        ax2.axvspan(8, 10, color='gray', alpha=0.15)
        ax2.axhline(v_d, color='gray', linestyle=':', alpha=0.5)
        ax2.set_title(f'N={N}: mean v_x')
        ax2.set_xlabel('time [s]'); ax2.set_ylabel('m/s')
        ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)

    # row 2: summary plots — recovery duration vs N, max wake vs N
    Ns = np.array(N_vals)
    ax = fig.add_subplot(gs[2, 0:2])
    for cond_name, color, marker in [('lane_based', 'C3', 's'),
                                     ('lane_less', 'C0', 'o')]:
        dur = np.array([m[1]['recovery_duration'] for m in summary[cond_name]])
        ax.plot(Ns, dur, color=color, marker=marker, label=cond_name)
    ax.set_xlabel('N (cars in corridor)')
    ax.set_ylabel('recovery duration after release [s]')
    ax.set_title('Time for wake to clear after brake released')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[2, 2:4])
    for cond_name, color, marker in [('lane_based', 'C3', 's'),
                                     ('lane_less', 'C0', 'o')]:
        mw = np.array([m[1]['max_wake'] for m in summary[cond_name]])
        ax.plot(Ns, mw, color=color, marker=marker, label=cond_name)
    ax.set_xlabel('N (cars in corridor)')
    ax.set_ylabel('max wake count')
    ax.set_title('Peak number of cars dragged below 0.9 v_d')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle('Exp C — lane-less vs lane-based: brake one car, watch the wake',
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig('exp_capacity.png', dpi=110, bbox_inches='tight')
    print('\nsaved exp_capacity.png')


if __name__ == '__main__':
    main()
