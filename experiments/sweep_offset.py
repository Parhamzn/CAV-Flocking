"""Investigation #3: off-center initial-y offset sweep.

Both flocks are 1xN single-row. They start at y_1 = y_mid - dy/2 and
y_2 = y_mid + dy/2. dy=0 is head-on (the canonical setup); large dy is
"pre-sorted lanes" where the flocks barely meet in y.

Measured per dy:
  * inter-flock min
  * mean tau-force magnitude on agent 0 over the run
  * tau engagement count: number of (i, r, t) triples where the tau gate
    was active (within d_c AND headings differ >= 90 deg)
  * deflection of each flock
  * wall_proximity

Run on a wider road (y_hi=24) so large offsets fit.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.flock_layouts import grid_formation
from flocking_lib.multi_flock_sim import run_multi_flock
from flocking_lib.control_tau import control_tau


def measure_tau_activity(q, p, flock_id, params):
    """Count (i, r, t) where the tau gate is active, and mean |f_tau| over time."""
    num_steps = q.shape[2]
    N = q.shape[0]
    gate_active = 0
    f_tau_mags = []
    d_c = params['d_c']
    for t in range(num_steps):
        qt, pt = q[:, :, t], p[:, :, t]
        for i in range(N):
            sp = np.linalg.norm(pt[i])
            if sp < 1e-9:
                continue
            dir_i = pt[i] / sp
            for r in range(N):
                if flock_id[r] == flock_id[i]:
                    continue
                diff_q = qt[i] - qt[r]
                if np.linalg.norm(diff_q) > d_c:
                    continue
                sr = np.linalg.norm(pt[r])
                if sr < 1e-9:
                    continue
                if np.dot(dir_i, pt[r] / sr) > 0:
                    continue
                gate_active += 1
            # also measure resulting tau force magnitude on agent i
            f_t = control_tau(i, qt, pt, flock_id, params)
            f_tau_mags.append(np.linalg.norm(f_t))
    return gate_active, float(np.mean(f_tau_mags))


def main():
    params = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,'d_c':40.0,'c1_t':0.0,'c2_t':0.08,
        'p_d_flock1': np.array([+10.0, 0.0]),
        'p_d_flock2': np.array([-10.0, 0.0]),
    }
    s = params['d_a']; v1 = params['p_d_flock1']; v2 = params['p_d_flock2']
    y_hi = 24.0; y_mid = 12.0
    N_each = 4

    offsets = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0]
    rows = []
    for dy in offsets:
        y1 = y_mid - dy / 2; y2 = y_mid + dy / 2
        # clamp into road
        y1 = max(params['d_b'] + 0.1, y1)
        y2 = min(y_hi - params['d_b'] - 0.1, y2)
        q1, p1 = grid_formation(1, N_each,
                                x_center=-20 - (N_each - 1) * s / 2,
                                y_center=y1, spacing=s, vel=v1)
        q2, p2 = grid_formation(1, N_each,
                                x_center=+20 + (N_each - 1) * s / 2,
                                y_center=y2, spacing=s, vel=v2)
        q, p, u, fid = run_multi_flock([(q1, p1), (q2, p2)], params,
                                        y_lo=0.0, y_hi=y_hi, T=12.0, TS=0.02)
        # metrics
        inter = np.inf
        for i in range(q.shape[0]):
            for j in range(i + 1, q.shape[0]):
                if fid[i] == fid[j]:
                    continue
                d = np.linalg.norm(q[i, :, :] - q[j, :, :], axis=0).min()
                if d < inter:
                    inter = d
        gate_active, mean_tau = measure_tau_activity(q, p, fid, params)
        # deflection per flock (signed)
        mask1 = fid == 1; mask2 = fid == 2
        defl1 = float(q[mask1, 1, -1].mean() - q[mask1, 1, 0].mean())
        defl2 = float(q[mask2, 1, -1].mean() - q[mask2, 1, 0].mean())
        rows.append((dy, inter, gate_active, mean_tau, defl1, defl2,
                     y1, y2))
        print(f'dy={dy:5.1f}m  y1={y1:5.2f} y2={y2:5.2f}  inter={inter:5.2f}m  '
              f'gate_active={gate_active:6d}  mean|τ|={mean_tau:5.2f} m/s²  '
              f'defl1={defl1:+5.2f}m  defl2={defl2:+5.2f}m')

    # Plots
    dys     = [r[0] for r in rows]
    inters  = [r[1] for r in rows]
    gates   = [r[2] for r in rows]
    means   = [r[3] for r in rows]
    defl1s  = [r[4] for r in rows]
    defl2s  = [r[5] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    ax = axes[0, 0]
    ax.plot(dys, inters, '-o', color='purple')
    ax.axhline(2.0, color='r', linestyle='--', alpha=0.4, label='car width 2 m')
    ax.set_xlabel('initial dy offset [m]'); ax.set_ylabel('inter-flock min [m]')
    ax.set_title('Inter-flock minimum distance')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(dys, gates, '-o', color='orange')
    ax.set_xlabel('initial dy offset [m]'); ax.set_ylabel('# active (i,r,t) gate triples')
    ax.set_title('Total τ-engagement')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(dys, means, '-o', color='teal')
    ax.set_xlabel('initial dy offset [m]'); ax.set_ylabel(r'mean $|\tau|$ over run [m/s$^2$]')
    ax.set_title('Mean τ-force magnitude')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(dys, defl1s, '-o', label='Flock 1 (starts below)')
    ax.plot(dys, defl2s, '-s', label='Flock 2 (starts above)')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('initial dy offset [m]'); ax.set_ylabel('signed Δy of flock centroid [m]')
    ax.set_title('Per-flock deflection')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle('Off-center initial-placement sweep (1×4 vs 1×4, y_hi=24 m)', y=0.995)
    fig.tight_layout()
    fig.savefig('sweep_offset.png', dpi=110, bbox_inches='tight')
    print('saved sweep_offset.png')


if __name__ == '__main__':
    main()
