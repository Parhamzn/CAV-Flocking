"""Driver: two opposing CAV flocks on a laneless freeway (port of flocking_main.m).

Control law per Week-2 methodology slides:
    u_i = f_alpha + f_beta + f_gamma + f_tau

Double-integrator dynamics. The tau-agent is the project's novel addition,
mediating inter-flock passage.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from flocking_lib.control_alpha import control_alpha
from flocking_lib.control_beta import control_beta
from flocking_lib.control_gamma import control_gamma
from flocking_lib.control_tau import control_tau


def main():
    # ------------------------------ Parameters --------------------------------
    params = {
        # sigma-norm smoothing
        'e':   0.1,
        # phi shape (a == b => symmetric attractive/repulsive)
        'a':   5,
        'b':   5,
        # alpha-agent (within-flock lattice)
        'd_a': 7,                          # desired neighbour spacing [m]
        'r_a': 1.2 * 7,                    # interaction radius [m]
        'h_a': 0.2,
        'c1_a': 5,
        'c2_a': 2 * np.sqrt(5),
        # beta-agent (road boundary avoidance) — wide zone, gentle taper, strong
        'd_b': 3.0,
        'h_b': 0.2,
        'c1_b': 200,
        'c2_b': 2 * np.sqrt(200),
        # gamma-agent (navigational feedback)
        'c_g': 1.5,
        # tau-agent (McKenzie 2012, eq. 6.4): lateral-deflection swap matrix,
        # gated on cooperation distance + anti-parallel-ish headings.
        # Velocity-only variant: c1_t = 0 disables the position term which
        # would otherwise cancel the deflection in the first half of slow
        # encounters. See control_tau.py docstring for the empirical study.
        'd_c':  30.0,                      # cooperation distance [m]
        'c1_t': 0.0,
        'c2_t': 0.08,
    }
    a_max = 9.0                            # m/s^2 — Week-2 deck

    # ------------------------------ Scenario ----------------------------------
    TS         = 0.02
    T          = 16
    num_steps  = int(round(T / TS)) + 1
    v_d        = 10.0
    N_per_flock = 4
    N          = 2 * N_per_flock
    y_lo, y_hi = 0.0, 14.4

    params['p_d_flock1'] = np.array([+v_d, 0.0])
    params['p_d_flock2'] = np.array([-v_d, 0.0])
    flock_id = np.concatenate([np.ones(N_per_flock, dtype=int),
                               2 * np.ones(N_per_flock, dtype=int)])

    # ------------------------------ State -------------------------------------
    q = np.zeros((N, 2, num_steps))
    p = np.zeros((N, 2, num_steps))
    u = np.zeros((N, 2, num_steps))

    # Stagger initial y between the two flocks so head-on encounters have a
    # non-zero lateral n_ij — gives the tau-agent something to deflect on.
    # McKenzie geometry: single-row flocks head-on at the same y.
    # x-spacing == d_a so alpha-lattice is at rest at t=0; tau deflects in y.
    # Far initial separation so tau has time to engage and split lanes before
    # the encounter at x=0.
    flock1_x = np.linspace(-80, -80 + (N_per_flock - 1) * params['d_a'], N_per_flock)
    q[:N_per_flock, 0, 0] = flock1_x
    q[:N_per_flock, 1, 0] = 7.2
    p[:N_per_flock, :, 0] = np.tile(params['p_d_flock1'], (N_per_flock, 1))

    flock2_x = np.linspace(80 - (N_per_flock - 1) * params['d_a'], 80, N_per_flock)
    q[N_per_flock:, 0, 0] = flock2_x
    q[N_per_flock:, 1, 0] = 7.2
    p[N_per_flock:, :, 0] = np.tile(params['p_d_flock2'], (N_per_flock, 1))

    # ------------------------------ Time loop ---------------------------------
    for t in range(num_steps - 1):
        qt = q[:, :, t]
        pt = p[:, :, t]
        ut = np.zeros((N, 2))
        for i in range(N):
            f_alpha = control_alpha(i, qt, pt, flock_id, params)
            f_beta  = control_beta (i, qt, pt, y_lo, y_hi, params)
            f_gamma = control_gamma(i, qt, pt, flock_id, params)
            f_tau   = control_tau  (i, qt, pt, flock_id, params)
            ut[i] = f_alpha + f_beta + f_gamma + f_tau
        # acceleration saturation
        for i in range(N):
            mag = np.linalg.norm(ut[i])
            if mag > a_max:
                ut[i] = a_max * ut[i] / mag
        u[:, :, t] = ut
        # semi-implicit Euler (double-integrator)
        p[:, :, t + 1] = pt + TS * ut
        q[:, :, t + 1] = qt + TS * p[:, :, t + 1]

    # ------------------------------ Plots -------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for i in range(N):
        xs = q[i, 0, :]
        ys = q[i, 1, :]
        col = 'b' if flock_id[i] == 1 else 'r'
        ax1.plot(xs, ys, color=col, linewidth=1)
        ax1.plot(xs[0],  ys[0],  marker='o', color=col, markersize=7)
        ax1.plot(xs[-1], ys[-1], marker='s', color=col, markersize=7, fillstyle='none')
    ax1.axhline(y_lo, color='k', linewidth=2)
    ax1.axhline(y_hi, color='k', linewidth=2)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title(r'$\alpha + \beta + \gamma + \tau$ flocking — opposing CAV flocks')
    ax1.set_aspect('equal')
    ax1.grid(True)

    t_vec = np.arange(num_steps) * TS
    fig2, axes = plt.subplots(2, 2, figsize=(10, 6))
    for i in range(N):
        col = 'b' if flock_id[i] == 1 else 'r'
        axes[0, 0].plot(t_vec, q[i, 0, :], color=col)
        axes[0, 1].plot(t_vec, q[i, 1, :], color=col)
        axes[1, 0].plot(t_vec, p[i, 0, :], color=col)
        axes[1, 1].plot(t_vec, p[i, 1, :], color=col)
    axes[0, 0].set_title('x(t)');   axes[0, 0].set_ylabel('m')
    axes[0, 1].set_title('y(t)');   axes[0, 1].set_ylabel('m')
    axes[1, 0].set_title('v_x(t)'); axes[1, 0].set_ylabel('m/s'); axes[1, 0].set_xlabel('t [s]')
    axes[1, 1].set_title('v_y(t)'); axes[1, 1].set_ylabel('m/s'); axes[1, 1].set_xlabel('t [s]')
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    xl = (q[:, 0, :].min() - 5, q[:, 0, :].max() + 5)
    ax3.set_xlim(xl); ax3.set_ylim(y_lo - 1, y_hi + 1)
    ax3.axhline(y_lo, color='k', linewidth=2)
    ax3.axhline(y_hi, color='k', linewidth=2)
    ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]')
    cols = ['b' if fid == 1 else 'r' for fid in flock_id]
    scat = ax3.scatter(q[:, 0, 0], q[:, 1, 0], c=cols, s=100)
    title = ax3.set_title('t = 0.00 s')

    def update(frame):
        scat.set_offsets(q[:, :, frame])
        title.set_text(f't = {frame * TS:.2f} s')
        return scat, title

    anim = manimation.FuncAnimation(
        fig3, update, frames=range(0, num_steps, 2), interval=50, blit=False
    )

    plt.show()
    return q, p, u


if __name__ == '__main__':
    main()
