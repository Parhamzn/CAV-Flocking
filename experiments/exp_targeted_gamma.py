"""Investigation #5: position-based gamma with externally assigned target bands.

Targets the compression mode (investigation #1). Add a position-feedback
term to gamma that pulls each flock toward an assigned target y. With
sensible target assignment, agents should glide into their lanes without
slamming into walls (which is what caused the compression).

Test scenario: D from exp_geometries — 2×4 vs 2×4 on y_hi=24.
Target bands: flock 1 → y=5, flock 2 → y=19 (just outside d_b=3 zones).

Sweep c_g_pos in {0, 0.5, 1, 2, 4}, with τ on and τ off, and report:
  * intra-flock min (does compression go away?)
  * inter-flock min (does avoidance still work?)
  * how close each flock gets to its target
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.flock_layouts import grid_formation
from flocking_lib.multi_flock_sim import run_multi_flock


def metrics(q, p, flock_id, y_hi, d_b, y_targets):
    N = q.shape[0]
    N1 = (flock_id == 1).sum()
    inter_min = np.inf
    intra_min = np.inf
    for t in range(q.shape[2]):
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(q[i, :, t] - q[j, :, t])
                if flock_id[i] != flock_id[j]:
                    if d < inter_min: inter_min = d
                else:
                    if d < intra_min: intra_min = d
    # final y deviation from target
    final_y1 = q[flock_id == 1, 1, -1].mean()
    final_y2 = q[flock_id == 2, 1, -1].mean()
    return dict(inter_min=inter_min, intra_min=intra_min,
                final_y1=final_y1, final_y2=final_y2,
                err1=abs(final_y1 - y_targets[1]),
                err2=abs(final_y2 - y_targets[2]))


def main():
    base_params = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,'d_c':40.0,'c2_t':0.08,
        'p_d_flock1': np.array([+10.0, 0.0]),
        'p_d_flock2': np.array([-10.0, 0.0]),
    }
    y_hi = 24.0; y_mid = 12.0
    y_targets = {1: 5.0, 2: 19.0}   # externally assigned bands
    s = base_params['d_a']
    v1, v2 = base_params['p_d_flock1'], base_params['p_d_flock2']

    # Initial conditions (centered on y_mid)
    q1, p1 = grid_formation(2, 4, x_center=-20 - 1.5*s, y_center=y_mid, spacing=s, vel=v1)
    q2, p2 = grid_formation(2, 4, x_center=+20 + 1.5*s, y_center=y_mid, spacing=s, vel=v2)

    sweep = [0.0, 0.5, 1.0, 2.0, 4.0]
    print(f'{"":>9}  | {"τ on (c1_t=0, c2_t=0.08)":>40}  | {"τ off (c2_t=0)":>40}')
    print(f'{"c_g_pos":>9}  | {"inter intra y1→5 y2→19":>40}  | {"inter intra y1→5 y2→19":>40}')
    rows = []
    for c_g_pos in sweep:
        line = []
        for tau_on in (True, False):
            params = dict(base_params,
                          c_g_pos=c_g_pos,
                          y_target_per_flock=y_targets,
                          c1_t=0.0,
                          c2_t=0.08 if tau_on else 0.0)
            q, p, u, fid = run_multi_flock([(q1, p1), (q2, p2)], params,
                                            y_lo=0.0, y_hi=y_hi, T=12.0, TS=0.02)
            m = metrics(q, p, fid, y_hi, params['d_b'], y_targets)
            line.append((m, q, p, fid))
        m_on, _, _, _ = line[0]; m_off, _, _, _ = line[1]
        print(f'{c_g_pos:>9.2f}  | inter={m_on["inter_min"]:5.2f} intra={m_on["intra_min"]:5.2f} '
              f'err1={m_on["err1"]:5.2f} err2={m_on["err2"]:5.2f}  '
              f'| inter={m_off["inter_min"]:5.2f} intra={m_off["intra_min"]:5.2f} '
              f'err1={m_off["err1"]:5.2f} err2={m_off["err2"]:5.2f}')
        rows.append((c_g_pos, line))

    # Plot trajectory for selected c_g_pos values
    selected = [0.0, 1.0, 4.0]
    fig, axes = plt.subplots(len(selected), 2, figsize=(14, 4*len(selected)))
    for row, c_g_pos in enumerate(selected):
        for col, tau_on in enumerate([True, False]):
            params = dict(base_params, c_g_pos=c_g_pos, y_target_per_flock=y_targets,
                          c1_t=0.0, c2_t=0.08 if tau_on else 0.0)
            q, p, u, fid = run_multi_flock([(q1, p1), (q2, p2)], params,
                                            y_lo=0.0, y_hi=y_hi, T=12.0, TS=0.02)
            m = metrics(q, p, fid, y_hi, params['d_b'], y_targets)
            ax = axes[row, col]
            N = q.shape[0]
            for i in range(N):
                xs, ys = q[i, 0, :], q[i, 1, :]
                color = 'b' if fid[i] == 1 else 'r'
                ax.plot(xs, ys, color=color, linewidth=1, alpha=0.7)
                ax.plot(xs[0], ys[0], marker='o', color=color, markersize=6)
                ax.plot(xs[-1], ys[-1], marker='s', color=color, markersize=6, fillstyle='none')
            ax.axhline(0, color='k', linewidth=1.5); ax.axhline(y_hi, color='k', linewidth=1.5)
            ax.axhline(y_targets[1], color='b', linestyle=':', alpha=0.5, label='flock 1 target')
            ax.axhline(y_targets[2], color='r', linestyle=':', alpha=0.5, label='flock 2 target')
            ax.set_xlim(q[:,0,:].min()-5, q[:,0,:].max()+5)
            ax.set_ylim(-1, y_hi+1)
            ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
            ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
            tag = 'τ ON' if tau_on else 'τ OFF'
            ax.set_title(f'c_g_pos={c_g_pos}  {tag}\n'
                         f'inter={m["inter_min"]:.2f}  intra={m["intra_min"]:.2f}',
                         fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle('Position-based γ on scenario D (2×4 vs 2×4, y_hi=24)', y=0.995)
    fig.tight_layout()
    fig.savefig('exp_targeted_gamma.png', dpi=110, bbox_inches='tight')
    print('saved exp_targeted_gamma.png')


if __name__ == '__main__':
    main()
