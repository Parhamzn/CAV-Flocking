"""Diagnose why low v_d causes the McKenzie tau-agent to fail.

Hypothesis: the position term c1*J*(q_i - q_r) is symmetric in time (pushes
up before crossing, down after, integrates to ~zero net deflection), while
the velocity term c2*J*(p_i - p_r) scales linearly with v_d. So at low v_d
the position term dominates over a velocity term that's too weak to win.

Test: run v_d=3 and v_d=10 side by side. For agent 0, log:
  * y position over time
  * y-component of the position-term tau contribution
  * y-component of the velocity-term tau contribution
  * y-component of the net tau force

Plot the two cases side by side.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.control_alpha import control_alpha
from flocking_lib.control_beta import control_beta
from flocking_lib.control_gamma import control_gamma


# Re-implement control_tau here so we can split its position vs velocity
# contributions into two return values rather than one summed vector.
J = np.array([[0.0, 1.0], [1.0, 0.0]])


def control_tau_split(i, q, p, flock_id, params):
    qi = q[i]; pi_v = p[i]
    f_pos = np.zeros(2); f_vel = np.zeros(2)
    sp = np.linalg.norm(pi_v)
    if sp < 1e-9:
        return f_pos, f_vel
    dir_i = pi_v / sp
    for r in range(q.shape[0]):
        if flock_id[r] == flock_id[i]:
            continue
        diff_q = qi - q[r]
        if np.linalg.norm(diff_q) > params['d_c']:
            continue
        sr = np.linalg.norm(p[r])
        if sr < 1e-9:
            continue
        if np.dot(dir_i, p[r] / sr) > 0:
            continue
        f_pos = f_pos - params['c1_t'] * (J @ diff_q)
        f_vel = f_vel - params['c2_t'] * (J @ (pi_v - p[r]))
    return f_pos, f_vel


def run(v_d, params, scenario):
    N_per_flock = scenario['N_per_flock']; N = 2 * N_per_flock
    TS = scenario['TS']; T = scenario['T']
    num_steps = int(round(T / TS)) + 1
    a_max = scenario['a_max']
    y_lo, y_hi = 0.0, scenario['y_hi']
    y_mid = 0.5 * (y_lo + y_hi)

    params = dict(params)
    params['p_d_flock1'] = np.array([+v_d, 0.0])
    params['p_d_flock2'] = np.array([-v_d, 0.0])

    flock_id = np.concatenate([np.ones(N_per_flock, dtype=int),
                               2 * np.ones(N_per_flock, dtype=int)])
    q = np.zeros((N, 2, num_steps))
    p = np.zeros((N, 2, num_steps))
    # diagnostics for agent 0 only
    f_pos_hist = np.zeros((num_steps, 2))
    f_vel_hist = np.zeros((num_steps, 2))
    f_a_hist   = np.zeros((num_steps, 2))
    f_b_hist   = np.zeros((num_steps, 2))
    f_g_hist   = np.zeros((num_steps, 2))

    x_inner = scenario['x_inner']; d_a = params['d_a']
    f1x = np.linspace(-x_inner - (N_per_flock - 1) * d_a, -x_inner, N_per_flock)
    f2x = np.linspace( x_inner,  x_inner + (N_per_flock - 1) * d_a, N_per_flock)
    q[:N_per_flock,  0, 0] = f1x; q[N_per_flock:, 0, 0] = f2x
    q[:, 1, 0] = y_mid
    p[:N_per_flock,  :, 0] = np.tile(params['p_d_flock1'], (N_per_flock, 1))
    p[N_per_flock:,  :, 0] = np.tile(params['p_d_flock2'], (N_per_flock, 1))

    for t in range(num_steps - 1):
        qt, pt = q[:, :, t], p[:, :, t]
        ut = np.zeros((N, 2))
        for i in range(N):
            f_a = control_alpha(i, qt, pt, flock_id, params)
            f_b = control_beta (i, qt, pt, y_lo, y_hi, params)
            f_g = control_gamma(i, qt, pt, flock_id, params)
            f_p, f_v = control_tau_split(i, qt, pt, flock_id, params)
            ut[i] = f_a + f_b + f_g + f_p + f_v
            if i == 3:
                f_a_hist[t] = f_a; f_b_hist[t] = f_b; f_g_hist[t] = f_g
                f_pos_hist[t] = f_p; f_vel_hist[t] = f_v
        for i in range(N):
            mag = np.linalg.norm(ut[i])
            if mag > a_max:
                ut[i] = a_max * ut[i] / mag
        p[:, :, t + 1] = pt + TS * ut
        q[:, :, t + 1] = qt + TS * p[:, :, t + 1]

    return dict(q=q, p=p, flock_id=flock_id,
                f_pos=f_pos_hist, f_vel=f_vel_hist,
                f_a=f_a_hist, f_b=f_b_hist, f_g=f_g_hist,
                TS=TS)


def main():
    params = {
        'e': 0.1, 'a': 5, 'b': 5,
        'd_a': 7, 'r_a': 1.2 * 7, 'h_a': 0.2,
        'c1_a': 5, 'c2_a': 2 * np.sqrt(5),
        'd_b': 3.0, 'h_b': 0.2,
        'c1_b': 200, 'c2_b': 2 * np.sqrt(200),
        'c_g': 1.5,
        'd_c': 30.0, 'c1_t': 0.02, 'c2_t': 0.08,
    }
    base = dict(TS=0.02, a_max=9.0, x_inner=20.0, y_hi=14.4, N_per_flock=4)

    cases = [
        ('v_d = 3 m/s (failure)',  3.0, dict(base, T=22.0)),
        ('v_d = 10 m/s (working)', 10.0, dict(base, T=11.0)),
    ]
    runs = [(label, v, run(v, params, sc)) for (label, v, sc) in cases]

    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    for col, (label, v_d, r) in enumerate(runs):
        q, TS = r['q'], r['TS']
        t_vec = np.arange(q.shape[2]) * TS

        # row 0: y trajectory of agent 0 (flock 1) and agent 4 (flock 2)
        ax = axes[0, col]
        ax.plot(t_vec, q[3, 1, :], 'b-', label='agent 3 (flock 1, innermost)')
        ax.plot(t_vec, q[4, 1, :], 'r-', label='agent 4 (flock 2, innermost)')
        ax.axhline(7.2, color='k', linestyle=':', alpha=0.5, label='start y')
        ax.set_ylabel('y [m]'); ax.set_title(f'{label} — y trajectory')
        ax.legend(fontsize=8); ax.grid(True)

        # row 1: tau force y-components for agent 0
        ax = axes[1, col]
        ax.plot(t_vec, r['f_pos'][:, 1], 'C0-', label=r'$\tau$ position term (y)')
        ax.plot(t_vec, r['f_vel'][:, 1], 'C1-', label=r'$\tau$ velocity term (y)')
        ax.plot(t_vec, r['f_pos'][:, 1] + r['f_vel'][:, 1], 'k-', linewidth=2,
                label=r'$\tau$ total (y)')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_ylabel('force [m/s²]'); ax.set_title(r'$\tau$-force decomposition (agent 3, innermost)')
        ax.legend(fontsize=8); ax.grid(True)

        # row 2: all force y-components for agent 0
        ax = axes[2, col]
        ax.plot(t_vec, r['f_a'][:, 1], label=r'$\alpha$')
        ax.plot(t_vec, r['f_b'][:, 1], label=r'$\beta$')
        ax.plot(t_vec, r['f_g'][:, 1], label=r'$\gamma$')
        ax.plot(t_vec, r['f_pos'][:, 1] + r['f_vel'][:, 1], 'k-', linewidth=2,
                label=r'$\tau$ (total)')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('t [s]'); ax.set_ylabel('force [m/s²]')
        ax.set_title('all y-forces on agent 3 (innermost)')
        ax.legend(fontsize=8); ax.grid(True)

    fig.tight_layout()
    fig.savefig('diagnose_slow_failure.png', dpi=110, bbox_inches='tight')
    print('saved diagnose_slow_failure.png')


if __name__ == '__main__':
    main()
