"""Investigation #4: 3+ flocks meeting at an intersection.

Open-arena geometry (β-walls placed far away so they don't activate).
Three configurations:
  · 3 flocks at 120° (symmetric Y intersection)
  · 4 flocks at 90°  (cross intersection)
  · 4 flocks at 90° with mild offset (one flock starts closer, breaks symmetry)

For each: trajectory plot, inter-flock min over all flock-pairs, deadlock
detection (any agent with |v| < 0.5*v_d for > 3s).
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.flock_layouts import grid_formation
from flocking_lib.multi_flock_sim import run_multi_flock


def make_flock_at_angle(N, angle_deg, dist_from_center, spacing, v_d, perp_jitter=0):
    """Build a single-row flock heading toward origin from `dist_from_center`
    away along direction `angle_deg`. Flock spans `spacing*(N-1)` perpendicular
    to its motion. perp_jitter offsets the flock laterally by that amount.
    """
    angle = np.deg2rad(angle_deg)
    # Position: distance dist_from_center away, in direction angle_deg
    # so that velocity (toward origin) is -dir(angle).
    dir_to_center = -np.array([np.cos(angle), np.sin(angle)])
    perp = np.array([-np.sin(angle), np.cos(angle)])  # 90° rotated
    center = np.array([np.cos(angle), np.sin(angle)]) * dist_from_center + perp * perp_jitter
    # Lay the row perpendicular to its motion
    q = np.zeros((N, 2))
    for k in range(N):
        q[k] = center + perp * (k - (N - 1) / 2) * spacing
    p = np.tile(dir_to_center * v_d, (N, 1))
    return q, p


def run_intersection(angles, params, scenario, perp_jitters=None):
    if perp_jitters is None:
        perp_jitters = [0.0] * len(angles)
    flock_inits = []
    p_d_dict = {}
    for k, (angle_deg, jit) in enumerate(zip(angles, perp_jitters), start=1):
        q, p = make_flock_at_angle(
            scenario['N_per_flock'], angle_deg,
            scenario['dist_from_center'], params['d_a'],
            scenario['v_d'], perp_jitter=jit,
        )
        flock_inits.append((q, p))
        # desired velocity: same as initial (toward origin)
        p_d_dict[k] = p[0].copy()
    params = dict(params, p_d_per_flock=p_d_dict)
    q, p, u, fid = run_multi_flock(flock_inits, params,
                                    y_lo=scenario['y_lo'], y_hi=scenario['y_hi'],
                                    T=scenario['T'], TS=scenario['TS'])
    return q, p, u, fid


def metrics_intersection(q, p, flock_id, TS, v_d):
    """inter-flock pair min, max time below 0.5*v_d (per-agent), deadlock flag."""
    N = q.shape[0]
    # pairwise inter-flock min over time
    inter_min = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            if flock_id[i] == flock_id[j]:
                continue
            d = np.linalg.norm(q[i, :, :] - q[j, :, :], axis=0).min()
            if d < inter_min:
                inter_min = d
    # deadlock: any agent with speed < 0.5*v_d for more than 3 seconds CONSECUTIVELY
    speeds = np.linalg.norm(p, axis=1)  # (N, num_steps)
    max_stall = 0.0
    for i in range(N):
        below = speeds[i] < 0.5 * v_d
        # find max consecutive True run
        run = 0; best_run = 0
        for b in below:
            if b:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0
        stall_s = best_run * TS
        if stall_s > max_stall:
            max_stall = stall_s
    return dict(inter_min=inter_min, max_stall_s=max_stall,
                deadlock=max_stall > 3.0)


def main():
    params = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,'d_c':40.0,'c1_t':0.0,'c2_t':0.08,
    }
    scenario = dict(N_per_flock=4, dist_from_center=40.0, v_d=10.0,
                    y_lo=-100.0, y_hi=+100.0,  # walls far away, won't activate
                    T=14.0, TS=0.02)

    configs = [
        ('A · 3 flocks at 120°',         [0, 120, 240],     None),
        ('B · 4 flocks at 90° (sym)',    [0, 90, 180, 270], None),
        ('C · 4 flocks at 90° (jitter)', [0, 90, 180, 270], [0, 0.5, 0, -0.5]),
    ]

    results = []
    for label, angles, jitters in configs:
        q, p, u, fid = run_intersection(angles, params, scenario, perp_jitters=jitters)
        m = metrics_intersection(q, p, fid, scenario['TS'], scenario['v_d'])
        results.append((label, q, p, fid, m))
        print(f'{label:38s}  inter={m["inter_min"]:5.2f}m  max_stall={m["max_stall_s"]:5.2f}s  '
              f'deadlock={"YES" if m["deadlock"] else "no"}')

    # 3 trajectory plots (one per config) plus a per-config color per flock
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    flock_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
    for k, (label, q, p, fid, m) in enumerate(results):
        ax = axes[k]
        N = q.shape[0]
        max_fid = int(fid.max())
        for i in range(N):
            col = flock_colors[(fid[i] - 1) % len(flock_colors)]
            xs, ys = q[i, 0, :], q[i, 1, :]
            ax.plot(xs, ys, color=col, linewidth=1, alpha=0.7)
            ax.plot(xs[0], ys[0], marker='o', color=col, markersize=6)
            ax.plot(xs[-1], ys[-1], marker='s', color=col, markersize=6, fillstyle='none')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        lim = max(abs(q[:, 0, :]).max(), abs(q[:, 1, :]).max()) + 5
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title(f'{label}\ninter={m["inter_min"]:.2f}m  max_stall={m["max_stall_s"]:.2f}s'
                     f'  {"DEADLOCK" if m["deadlock"] else "passed"}',
                     fontsize=10)
    fig.suptitle('Intersection scenarios — open arena, no walls', fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig('exp_intersection.png', dpi=110, bbox_inches='tight')
    print('saved exp_intersection.png')


if __name__ == '__main__':
    main()
