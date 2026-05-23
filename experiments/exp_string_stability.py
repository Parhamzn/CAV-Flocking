"""Experiment F: string stability under leader perturbation.

For a platoon of N cars at v_d, brake the leader for 2 s then release.
For each car i downstream, compute the L2 norm of the velocity error
e_i(t) = v_i(t) - v_d over the perturbation window. Plot ||e_i|| vs car
index. A string-stable platoon has monotonically decreasing ||e_i|| as
i grows; an unstable one shows amplification.

Compared conditions:
  lane_locked : each car pinned to its initial y. Pure 1-D
                car-following through the α-lattice.
  lane_less   : full 2-D flocking. Cars may shift laterally to absorb
                the disturbance.

The platoon is laid out in a single file at x = 0, d_a, 2 d_a, ...
within a finite corridor (long enough that wrap-around isn't a factor
during the run). All cars start at the centre of the usable y-strip.

If lane-less is genuinely safer, we expect: lane-less ||e_i|| decays
faster, AND its peak ||e|| value is lower because lateral spreading
dissipates the brake energy across the y-axis.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import exp_fundamental_diagram as A


def _run_platoon(N, lane_locked, L=600.0, W=14.0, v_d=10.0,
                 T=30.0, TS=0.02,
                 t_brake_start=5.0, t_brake_end=7.0, v_brake=2.0,
                 seed=42):
    rng = np.random.default_rng(seed)
    q = np.zeros((N, 2))
    q[:, 0] = np.arange(N) * A.D_A + 50.0  # platoon laid out from x=50 m
    q[:, 1] = (A.D_B + (W - 2 * A.D_B) / 2) + rng.normal(0, 0.05, N)
    p = np.tile(np.array([v_d, 0.0]), (N, 1))
    y0 = q[:, 1].copy()

    num_steps = int(round(T / TS))
    vx_hist = np.zeros((num_steps, N))
    vy_hist = np.zeros((num_steps, N))
    y_hist = np.zeros((num_steps, N))

    leader = N - 1  # rightmost = head of platoon (going +x)

    for t in range(num_steps):
        time = t * TS
        u = A.alpha_force_periodic(q, p, L)
        u += A.beta_force_corridor(q, p, W)
        u += A.gamma_force(p, v_d)
        mags = np.linalg.norm(u, axis=1)
        scale = np.where(mags > A.A_MAX, A.A_MAX / np.maximum(mags, 1e-9), 1.0)
        u *= scale[:, None]
        if t_brake_start <= time < t_brake_end:
            p[leader, 0] = v_brake
            p[leader, 1] = 0.0
            u[leader] = 0.0
        if lane_locked:
            u[:, 1] = 0.0
            p[:, 1] = 0.0
            q[:, 1] = y0
        p += TS * u
        q += TS * p
        # no periodic wrap — platoon is short relative to L
        vx_hist[t] = p[:, 0]
        vy_hist[t] = p[:, 1]
        y_hist[t] = q[:, 1]

    return dict(vx=vx_hist, vy=vy_hist, y=y_hist, TS=TS, T=T,
                t_brake_start=t_brake_start, t_brake_end=t_brake_end,
                v_d=v_d, leader=leader)


def _disturbance_norms(run):
    """L2 norm of (v_x - v_d) per car, post-brake-onset, normalised to s."""
    TS = run['TS']
    t_axis = np.arange(run['vx'].shape[0]) * TS
    measure_mask = t_axis >= run['t_brake_start']
    e = run['vx'][measure_mask] - run['v_d']  # (T, N)
    norms = np.sqrt((e ** 2).sum(axis=0) * TS)  # L2 norm of e in time
    peak_abs_dev = np.abs(e).max(axis=0)
    return norms, peak_abs_dev


def main():
    N = 16
    print(f'Platoon of N={N} cars (leader = idx {N-1}, head of column going +x)\n')
    runs = {}
    for label, locked in [('lane_locked', True), ('lane_less', False)]:
        runs[label] = _run_platoon(N, lane_locked=locked)

    norms = {k: _disturbance_norms(v) for k, v in runs.items()}
    print(f'{"car idx":>8}  {"||e|| lane_locked":>18}  {"||e|| lane_less":>16}  {"ratio":>6}')
    print(f'{"(0=tail)":>8}  {"L2 norm of v_x-v_d":>18}  {"":>16}  {"lk/ll":>6}')
    for i in range(N):
        n_lk, _ = norms['lane_locked']
        n_ll, _ = norms['lane_less']
        ratio = n_lk[i] / max(n_ll[i], 1e-6)
        print(f'{i:8d}  {n_lk[i]:18.3f}  {n_ll[i]:16.3f}  {ratio:6.2f}')

    # ---- string-stability classification --------------------------------
    # Walk from leader (perturbed source) backwards through the platoon.
    # String-stable ⟺ ||e_{i-1}|| ≤ ||e_i|| at every step (disturbance never
    # grows as it propagates downstream). The leader-to-first-follower
    # transition is the diagnostic one: if that step grows, the platoon
    # amplifies its own input.
    print('\nDownstream propagation (leader → tail):')
    for label in ['lane_locked', 'lane_less']:
        n_arr, _ = norms[label]
        leader = runs[label]['leader']
        path = [n_arr[leader - k] for k in range(leader + 1)]
        leader_to_first = path[1] / max(path[0], 1e-6)
        max_grow = max((path[k+1] / max(path[k], 1e-6)
                        for k in range(len(path)-1)), default=1.0)
        is_stable = max_grow <= 1.0 + 1e-3
        verdict = "STRING-STABLE" if is_stable else "AMPLIFIES"
        print(f"  {label:>11}: leader→follower ratio = {leader_to_first:.2f}, "
              f"max growth ratio anywhere = {max_grow:.2f}  → {verdict}")

    # ---- plot ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))

    ax = axes[0, 0]
    car_idx = np.arange(N)
    for label, color, marker in [('lane_locked', 'C3', 's'),
                                 ('lane_less', 'C0', 'o')]:
        n_arr, _ = norms[label]
        ax.plot(car_idx, n_arr, color=color, marker=marker, label=label)
    ax.set_xlabel('car index (0 = tail, N-1 = leader/braked)')
    ax.set_ylabel('||v_x - v_d||_2  [m/s · √s]')
    ax.set_title('Disturbance magnitude per car — does it decay or amplify?')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.axvline(N - 1, color='gray', linestyle=':', alpha=0.7,
               label='leader (braked)')

    ax = axes[0, 1]
    for label, color, marker in [('lane_locked', 'C3', 's'),
                                 ('lane_less', 'C0', 'o')]:
        _, peak = norms[label]
        ax.plot(car_idx, peak, color=color, marker=marker, label=label)
    ax.set_xlabel('car index')
    ax.set_ylabel('peak |v_x - v_d|  [m/s]')
    ax.set_title('Peak velocity deviation per car')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    t_axis = np.arange(runs['lane_locked']['vx'].shape[0]) * runs['lane_locked']['TS']
    cmap = plt.get_cmap('plasma')
    for i in range(N):
        ax.plot(t_axis, runs['lane_locked']['vx'][:, i],
                color=cmap(i / (N - 1)), linewidth=0.9)
    ax.axvspan(runs['lane_locked']['t_brake_start'],
               runs['lane_locked']['t_brake_end'],
               color='gray', alpha=0.15, label='brake')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('v_x [m/s]')
    ax.set_title('lane_locked: v_x(t) per car (dark = tail, bright = leader)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for i in range(N):
        ax.plot(t_axis, runs['lane_less']['vx'][:, i],
                color=cmap(i / (N - 1)), linewidth=0.9)
    ax.axvspan(runs['lane_less']['t_brake_start'],
               runs['lane_less']['t_brake_end'],
               color='gray', alpha=0.15)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('v_x [m/s]')
    ax.set_title('lane_less: v_x(t) per car')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Exp F — string stability of a {N}-car platoon under '
                 f'2 s leader brake', y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig('exp_string_stability.png', dpi=110, bbox_inches='tight')
    print('\nsaved exp_string_stability.png')


if __name__ == '__main__':
    main()
