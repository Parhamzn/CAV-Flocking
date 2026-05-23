"""Diagnose the multi-row flock compression in scenario D (2x4 vs 2x4).

Intra-flock minimum dropped from d_a=7 m at t=0 to ~4.3 m during the encounter.
This script:
  * runs the 2x4 vs 2x4 case
  * traces every intra-flock pair distance over time so we can see WHICH pair
    is getting squeezed and WHEN
  * traces y of the "front row" and "back row" of flock 1 to see if it's
    row-to-row compression (vertical) or column-to-column (horizontal)
  * traces the alpha-, beta- and tau-force y-components on a representative
    inner agent so we can attribute the squeeze to one of them
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.flock_layouts import grid_formation
from flocking_lib.multi_flock_sim import run_multi_flock
from flocking_lib.control_alpha import control_alpha
from flocking_lib.control_beta import control_beta
from flocking_lib.control_gamma import control_gamma
from flocking_lib.control_tau import control_tau


def run_with_force_trace(flock_inits, params, y_lo, y_hi, T, TS,
                         a_max=9.0, trace_agent=0):
    """Same as run_multi_flock but logs per-agent force components for one
    chosen agent at every step.
    """
    num_steps = int(round(T / TS)) + 1
    N_total = sum(q.shape[0] for q, _ in flock_inits)
    q = np.zeros((N_total, 2, num_steps))
    p = np.zeros((N_total, 2, num_steps))
    flock_id = np.zeros(N_total, dtype=int)
    offset = 0
    for k, (qi, pi) in enumerate(flock_inits, start=1):
        n = qi.shape[0]
        q[offset:offset + n, :, 0] = qi
        p[offset:offset + n, :, 0] = pi
        flock_id[offset:offset + n] = k
        offset += n

    f_a_hist = np.zeros((num_steps, 2))
    f_b_hist = np.zeros((num_steps, 2))
    f_g_hist = np.zeros((num_steps, 2))
    f_t_hist = np.zeros((num_steps, 2))

    for t in range(num_steps - 1):
        qt, pt = q[:, :, t], p[:, :, t]
        ut = np.zeros((N_total, 2))
        for i in range(N_total):
            f_a = control_alpha(i, qt, pt, flock_id, params)
            f_b = control_beta (i, qt, pt, y_lo, y_hi, params)
            f_g = control_gamma(i, qt, pt, flock_id, params)
            f_t = control_tau  (i, qt, pt, flock_id, params)
            ut[i] = f_a + f_b + f_g + f_t
            if i == trace_agent:
                f_a_hist[t] = f_a; f_b_hist[t] = f_b
                f_g_hist[t] = f_g; f_t_hist[t] = f_t
        for i in range(N_total):
            mag = np.linalg.norm(ut[i])
            if mag > a_max:
                ut[i] = a_max * ut[i] / mag
        p[:, :, t + 1] = pt + TS * ut
        q[:, :, t + 1] = qt + TS * p[:, :, t + 1]
    return dict(q=q, p=p, flock_id=flock_id, TS=TS,
                f_a=f_a_hist, f_b=f_b_hist, f_g=f_g_hist, f_t=f_t_hist)


def main():
    params = {
        'e':0.1,'a':5,'b':5,
        'd_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,
        'd_c':40.0,'c1_t':0.0,'c2_t':0.08,
        'p_d_flock1': np.array([+10.0, 0.0]),
        'p_d_flock2': np.array([-10.0, 0.0]),
    }
    s = params['d_a']
    v1 = params['p_d_flock1']; v2 = params['p_d_flock2']

    # Scenario D: 2x4 vs 2x4, both centered at y=12 on a 24 m road
    q1, p1 = grid_formation(2, 4, x_center=-20 - 1.5*s, y_center=12.0, spacing=s, vel=v1)
    q2, p2 = grid_formation(2, 4, x_center=+20 + 1.5*s, y_center=12.0, spacing=s, vel=v2)

    # Layout of flock 1 (rows = inner index, cols = outer index in grid_formation):
    # agent 0 = (col 0, row 0) = leftmost-back
    # agent 1 = (col 1, row 0)
    # agent 2 = (col 2, row 0)
    # agent 3 = (col 3, row 0) = rightmost-back (innermost relative to encounter)
    # agent 4 = (col 0, row 1) = leftmost-front
    # ...
    # agent 7 = (col 3, row 1) = rightmost-front
    #
    # grid_formation builds with meshgrid(xs, ys), so X.ravel() gives row-major
    # order. Let me verify:
    print('Flock 1 initial positions:')
    for i in range(8):
        print(f'  agent {i}: ({q1[i,0]:6.2f}, {q1[i,1]:6.2f})')

    # Run with force trace on agent 3 (rightmost car of one of the rows — the
    # one closest to the encounter point)
    r = run_with_force_trace(
        [(q1, p1), (q2, p2)], params, y_lo=0.0, y_hi=24.0, T=12.0, TS=0.02,
        trace_agent=3,
    )
    q, p, flock_id, TS = r['q'], r['p'], r['flock_id'], r['TS']
    t_vec = np.arange(q.shape[2]) * TS

    # ---- intra-flock distance traces for flock 1 (agents 0-7) ---------------
    N1 = 8
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    for i in range(N1):
        for j in range(i + 1, N1):
            d = np.linalg.norm(q[i, :, :] - q[j, :, :], axis=0)
            ax.plot(t_vec, d, alpha=0.4, linewidth=0.8)
    ax.axhline(7, color='k', linestyle=':', alpha=0.5, label=r'$d_a = 7$')
    ax.set_ylabel('pairwise dist within flock 1 [m]')
    ax.set_title('All intra-flock-1 pair distances')
    ax.set_xlabel('t [s]'); ax.legend(); ax.grid(True, alpha=0.3)

    # ---- y of each row of flock 1 over time --------------------------------
    ax = axes[0, 1]
    # In grid_formation(rows=2, cols=4), X.ravel() yields row-major: row 0 first
    # row 0 = agents 0-3 (y_center - spacing/2 = 12 - 3.5 = 8.5)
    # row 1 = agents 4-7 (y_center + spacing/2 = 12 + 3.5 = 15.5)
    for i in range(N1):
        col_c = 'C0' if i < 4 else 'C3'  # row 0 blue, row 1 red
        label = 'row 0 (lower)' if i == 0 else ('row 1 (upper)' if i == 4 else None)
        ax.plot(t_vec, q[i, 1, :], color=col_c, alpha=0.7, label=label)
    ax.axhline(0, color='k', linewidth=1); ax.axhline(24, color='k', linewidth=1)
    ax.set_ylabel('y [m]'); ax.set_title('Flock 1: y over time (row 0 vs row 1)')
    ax.set_xlabel('t [s]'); ax.legend(); ax.grid(True, alpha=0.3)

    # ---- force decomposition on agent 3 -----------------------------------
    ax = axes[1, 0]
    ax.plot(t_vec, r['f_a'][:, 1], label=r'$\alpha$', linewidth=1.2)
    ax.plot(t_vec, r['f_b'][:, 1], label=r'$\beta$', linewidth=1.2)
    ax.plot(t_vec, r['f_g'][:, 1], label=r'$\gamma$', linewidth=1.2)
    ax.plot(t_vec, r['f_t'][:, 1], 'k-', label=r'$\tau$', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel(r'y-force on agent 3 [m/s$^2$]')
    ax.set_title('Force decomposition (agent 3, lower row, innermost)')
    ax.set_xlabel('t [s]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ---- vertical row separation over time ---------------------------------
    ax = axes[1, 1]
    # mean y of row 0 (agents 0-3) vs row 1 (agents 4-7)
    y_row0 = q[:4, 1, :].mean(axis=0)
    y_row1 = q[4:8, 1, :].mean(axis=0)
    ax.plot(t_vec, y_row1 - y_row0, color='purple', linewidth=2)
    ax.axhline(7, color='k', linestyle=':', alpha=0.5, label='initial row spacing d_a=7')
    ax.set_ylabel('row 1 mean y  -  row 0 mean y [m]')
    ax.set_title('Vertical row separation (flock 1)')
    ax.set_xlabel('t [s]'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('Compression diagnostic — 2x4 vs 2x4 on y_hi=24', y=0.995)
    fig.tight_layout()
    fig.savefig('diagnose_compression.png', dpi=110, bbox_inches='tight')
    print('saved diagnose_compression.png')


if __name__ == '__main__':
    main()
