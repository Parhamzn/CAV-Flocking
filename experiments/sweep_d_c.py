"""Sweep cooperation distance d_c with the corrected algorithm
(velocity-only tau + fixed beta-agent).

d_c is the radius at which the tau-agent's binary gate sigma_2 turns on:
   sigma_2 = 1 iff  |q_i - q_r| <= d_c  AND  headings differ by >= 90 deg.

Hypothesis: with velocity-only tau, the force magnitude per pair is
independent of separation; what matters is the total time tau is engaged.
Larger d_c means earlier engagement and longer engagement window, so
inter-flock min should grow with d_c, then saturate when the deflection
budget is no longer the limiting factor.

Geometry held constant: N=4 per flock, single-row head-on, x_inner=20,
y_hi=14.4, v_d=10 m/s.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from sweep_v_d import run_sim, metrics


def main():
    params_template = {
        'e':0.1, 'a':5, 'b':5,
        'd_a':7, 'r_a':1.2*7, 'h_a':0.2, 'c1_a':5, 'c2_a':2*np.sqrt(5),
        'd_b':3.0, 'h_b':0.2, 'c1_b':200, 'c2_b':2*np.sqrt(200),
        'c_g':1.5,
        'c1_t':0.0,         # velocity-only tau (the fix)
        'c2_t':0.08,
    }
    scenario = dict(TS=0.02, T=11.0, a_max=9.0, x_inner=20.0,
                    y_hi=14.4, N_per_flock=4)
    v_d = 10.0

    d_cs = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 70.0, 100.0]
    rows = []
    for d_c in d_cs:
        params = dict(params_template, d_c=d_c)
        q,p,fid = run_sim(v_d, params, scenario)
        m = metrics(q, fid, scenario['y_hi'], params['d_b'])
        rows.append((d_c, m))
        print(f'd_c={d_c:6.1f}m  inter={m["inter_min"]:5.2f}m  '
              f'intra={m["intra_min"]:5.2f}m  wall%={m["wall_proximity_fraction"]:.2f}  '
              f'escapes={m["escapes"]:4d}  clearance@x0={m["clearance_at_x0"]:5.2f}m')

    ds   = [r[0] for r in rows]
    inter = [r[1]['inter_min'] for r in rows]
    intra = [r[1]['intra_min'] for r in rows]
    clear0 = [r[1]['clearance_at_x0'] for r in rows]
    wall = [r[1]['wall_proximity_fraction'] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ds, inter, '-o', label='inter-flock min')
    axes[0].plot(ds, intra, '-s', label='intra-flock min')
    axes[0].plot(ds, clear0, '-^', label='clearance at x-encounter')
    axes[0].axhline(2.0, color='r', linestyle='--', alpha=0.4, label='car width 2 m')
    axes[0].set_xlabel('cooperation distance d_c [m]')
    axes[0].set_ylabel('distance [m]')
    axes[0].set_title('Separation vs d_c (velocity-only τ, v_d=10)')
    axes[0].legend(fontsize=8); axes[0].grid(True)

    axes[1].plot(ds, wall, '-o', color='purple')
    axes[1].set_xlabel('d_c [m]')
    axes[1].set_ylabel('fraction of (agent x step)')
    axes[1].set_title(r'Time within $d_b$ of wall')
    axes[1].grid(True)

    fig.suptitle('Cooperation distance sweep — fixed algorithm')
    fig.tight_layout()
    fig.savefig('sweep_d_c.png', dpi=110, bbox_inches='tight')
    print('saved sweep_d_c.png')


if __name__ == '__main__':
    main()
