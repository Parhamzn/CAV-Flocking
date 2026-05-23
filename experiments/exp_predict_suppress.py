"""Investigation #6: predictive gating used to SUPPRESS τ (not trigger).

Idea: McKenzie's original distance+heading gate stays in place. Additionally,
if the projected closest-approach distance of an (i, r) pair exceeds a
threshold, suppress τ for that pair — these cars would have passed safely
without intervention.

This should:
  * Eliminate the "wasted τ" at large initial y-offset (investigation #3)
  * Not regress the head-on case (investigation #1's canonical scenario)
  * Possibly reduce wall-pressure from over-deflection

Threshold choice: a few car-widths (we try 3, 6, 10, and 20 m).
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.flock_layouts import grid_formation
from flocking_lib.multi_flock_sim import run_multi_flock, encounter_metrics
from flocking_lib.control_tau import control_tau


def count_tau_engagement(q, p, flock_id, params):
    """Count (i, r, t) triples where the FULL gate (incl. suppress) is open."""
    num = 0
    d_c = params['d_c']
    suppress_th = params.get('predict_suppress_threshold')
    for t in range(q.shape[2]):
        qt = q[:, :, t]; pt = p[:, :, t]
        for i in range(qt.shape[0]):
            sp = np.linalg.norm(pt[i])
            if sp < 1e-9: continue
            dir_i = pt[i] / sp
            for r in range(qt.shape[0]):
                if flock_id[r] == flock_id[i]: continue
                diff_q = qt[i] - qt[r]
                if np.linalg.norm(diff_q) > d_c: continue
                sr = np.linalg.norm(pt[r])
                if sr < 1e-9: continue
                if np.dot(dir_i, pt[r] / sr) > 0: continue
                if suppress_th is not None:
                    diff_p = pt[i] - pt[r]
                    denom = float(np.dot(diff_p, diff_p))
                    if denom > 1e-9:
                        t_star = max(0.0, -float(np.dot(diff_q, diff_p)) / denom)
                    else:
                        t_star = 0.0
                    if np.linalg.norm(diff_q + t_star * diff_p) > suppress_th:
                        continue
                num += 1
    return num


def main():
    base = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,'d_c':40.0,'c1_t':0.0,'c2_t':0.08,
        'p_d_flock1': np.array([+10.0, 0.0]),
        'p_d_flock2': np.array([-10.0, 0.0]),
    }
    s = base['d_a']
    v1, v2 = base['p_d_flock1'], base['p_d_flock2']
    y_hi = 24.0; y_mid = 12.0
    N_each = 4

    # Run the offset sweep with several suppression thresholds
    offsets = [0.0, 2.0, 4.0, 8.0, 14.0]
    thresholds = [None, 3.0, 6.0, 10.0, 20.0]

    print(f'\n{"dy":>5} | ' + ' | '.join(f'th={"none" if t is None else t}'.rjust(20) for t in thresholds))
    print('     | ' + ' | '.join(f'{"inter / gate-count":>20}' for _ in thresholds))
    rows = []
    for dy in offsets:
        y1 = max(base['d_b']+0.1, y_mid - dy/2); y2 = min(y_hi-base['d_b']-0.1, y_mid + dy/2)
        q1, p1 = grid_formation(1, N_each, x_center=-20-(N_each-1)*s/2,
                                y_center=y1, spacing=s, vel=v1)
        q2, p2 = grid_formation(1, N_each, x_center=+20+(N_each-1)*s/2,
                                y_center=y2, spacing=s, vel=v2)
        line_results = []
        for th in thresholds:
            params = dict(base)
            if th is not None:
                params['predict_suppress_threshold'] = th
            q, p, u, fid = run_multi_flock([(q1, p1), (q2, p2)], params,
                                            y_lo=0.0, y_hi=y_hi, T=12.0, TS=0.02)
            m = encounter_metrics(q, fid, y_hi, params['d_b'])
            gate_count = count_tau_engagement(q, p, fid, params)
            line_results.append((m['inter_min'], gate_count))
        rows.append((dy, line_results))
        cells = ' | '.join(f'{im:5.2f}m / {gc:5d}'.rjust(20) for im, gc in line_results)
        print(f'{dy:5.1f} | {cells}')

    # Plot inter and gate-count vs threshold for each offset
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    th_labels = ['off' if t is None else f'{t}' for t in thresholds]
    th_x = np.arange(len(thresholds))
    for k, (dy, line) in enumerate(rows):
        inters = [im for im, _ in line]
        gates = [gc for _, gc in line]
        axes[0].plot(th_x, inters, '-o', label=f'dy={dy} m')
        axes[1].plot(th_x, gates,  '-o', label=f'dy={dy} m')
    axes[0].axhline(2.0, color='r', linestyle='--', alpha=0.4, label='car width')
    axes[0].set_xticks(th_x); axes[0].set_xticklabels(th_labels)
    axes[0].set_xlabel('predictive suppress threshold [m]')
    axes[0].set_ylabel('inter-flock min [m]')
    axes[0].set_title('Inter-flock min vs suppression threshold')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].set_xticks(th_x); axes[1].set_xticklabels(th_labels)
    axes[1].set_xlabel('predictive suppress threshold [m]')
    axes[1].set_ylabel('τ-engagement count (incl. suppress)')
    axes[1].set_title('τ-engagement count vs threshold')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    fig.suptitle('Predictive suppression on the offset sweep (1×4 vs 1×4)', y=0.99)
    fig.tight_layout()
    fig.savefig('exp_predict_suppress.png', dpi=110, bbox_inches='tight')
    print('saved exp_predict_suppress.png')


if __name__ == '__main__':
    main()
