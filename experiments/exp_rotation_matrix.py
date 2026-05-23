"""Investigation #7: replace J with R(+90°) — true rotation matrix.

McKenzie's J=[[0,1],[1,0]] is a reflection. For axis-aligned 1D flocks
(±x or ±y) it produces correct perpendicular deflection, but for arbitrary
flock orientations the J-deflection direction depends on the absolute
heading rather than relative geometry. This causes symmetric 3- and 4-flock
intersection scenarios to fail (investigation #4).

Hypothesis: R(+90°)=[[0,-1],[1,0]] gives every flock a consistent
"right-of-motion" deflection regardless of its heading direction, which
should produce roundabout-like passage at intersections.

Three tests:
   1. Canonical 2-flock opposing (1×4 vs 1×4 head-on). Should be UNCHANGED
      because both matrices give the same result for axis-aligned diff_p.
   2. 3-flock at 120°.
   3. 4-flock at 90° symmetric — the one that completely failed.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.flock_layouts import grid_formation
from flocking_lib.multi_flock_sim import run_multi_flock, encounter_metrics
from flocking_lib.control_tau import J, R_PLUS_90, R_MINUS_90
from exp_intersection import make_flock_at_angle, metrics_intersection


def run_canonical(M, params, scenario):
    s = params['d_a']; v_d = 10.0
    v1 = np.array([+v_d, 0.0]); v2 = np.array([-v_d, 0.0])
    q1, p1 = grid_formation(1, 4, x_center=-30.5, y_center=7.2, spacing=s, vel=v1)
    q2, p2 = grid_formation(1, 4, x_center=+30.5, y_center=7.2, spacing=s, vel=v2)
    params = dict(params, tau_matrix=M,
                  p_d_per_flock={1: v1, 2: v2})
    q, p, u, fid = run_multi_flock([(q1, p1), (q2, p2)], params,
                                    y_lo=0.0, y_hi=14.4, T=12.0, TS=0.02)
    return q, p, fid


def run_intersection_with_M(angles, M, params, scenario):
    flock_inits = []
    p_d_dict = {}
    for k, angle_deg in enumerate(angles, start=1):
        q, p = make_flock_at_angle(scenario['N_per_flock'], angle_deg,
                                   scenario['dist_from_center'],
                                   params['d_a'], scenario['v_d'])
        flock_inits.append((q, p))
        p_d_dict[k] = p[0].copy()
    p = dict(params, p_d_per_flock=p_d_dict, tau_matrix=M)
    q, pp, u, fid = run_multi_flock(flock_inits, p,
                                     y_lo=scenario['y_lo'], y_hi=scenario['y_hi'],
                                     T=scenario['T'], TS=scenario['TS'])
    return q, pp, fid


def main():
    params = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,'d_c':40.0,'c1_t':0.0,'c2_t':0.08,
    }
    sc = dict(N_per_flock=4, dist_from_center=40.0, v_d=10.0,
              y_lo=-100.0, y_hi=+100.0, T=14.0, TS=0.02)

    # ---- Test 1: canonical 2-flock (regression check) ---------------------
    print('=== Test 1: 2-flock head-on (verify no regression) ===')
    for label, M in [('J (McKenzie)', J), ('R(+90°)', R_PLUS_90), ('R(-90°)', R_MINUS_90)]:
        q, p, fid = run_canonical(M, params, None)
        m = encounter_metrics(q, fid, 14.4, params['d_b'])
        print(f'  {label:15s}  inter={m["inter_min"]:5.2f}m  intra={m["intra_min"]:5.2f}m')

    # ---- Test 2: 3-flock 120° --------------------------------------------
    print('\n=== Test 2: 3-flock at 120° ===')
    for label, M in [('J (McKenzie)', J), ('R(+90°)', R_PLUS_90), ('R(-90°)', R_MINUS_90)]:
        q, p, fid = run_intersection_with_M([0, 120, 240], M, params, sc)
        m = metrics_intersection(q, p, fid, sc['TS'], sc['v_d'])
        print(f'  {label:15s}  inter={m["inter_min"]:5.2f}m  max_stall={m["max_stall_s"]:.2f}s')

    # ---- Test 3: 4-flock 90° symmetric ----------------------------------
    print('\n=== Test 3: 4-flock at 90° symmetric ===')
    for label, M in [('J (McKenzie)', J), ('R(+90°)', R_PLUS_90), ('R(-90°)', R_MINUS_90)]:
        q, p, fid = run_intersection_with_M([0, 90, 180, 270], M, params, sc)
        m = metrics_intersection(q, p, fid, sc['TS'], sc['v_d'])
        print(f'  {label:15s}  inter={m["inter_min"]:5.2f}m  max_stall={m["max_stall_s"]:.2f}s')

    # ---- Trajectory comparison plot for the 4-flock case ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cols = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
    for col_idx, (label, M) in enumerate([
        ('J (McKenzie)', J), ('R(+90°)', R_PLUS_90), ('R(-90°)', R_MINUS_90)
    ]):
        q, p, fid = run_intersection_with_M([0, 90, 180, 270], M, params, sc)
        m = metrics_intersection(q, p, fid, sc['TS'], sc['v_d'])
        ax = axes[col_idx]
        for i in range(q.shape[0]):
            c = cols[(fid[i] - 1) % len(cols)]
            xs, ys = q[i, 0, :], q[i, 1, :]
            ax.plot(xs, ys, color=c, linewidth=1, alpha=0.7)
            ax.plot(xs[0],  ys[0],  marker='o', color=c, markersize=6)
            ax.plot(xs[-1], ys[-1], marker='s', color=c, markersize=6, fillstyle='none')
        lim = max(abs(q[:, 0, :]).max(), abs(q[:, 1, :]).max()) + 5
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title(f'{label}  inter={m["inter_min"]:.2f}m', fontsize=11)
    fig.suptitle('4-flock 90° intersection: J vs R(±90°)', y=0.995)
    fig.tight_layout()
    fig.savefig('exp_rotation_matrix.png', dpi=110, bbox_inches='tight')
    print('\nsaved exp_rotation_matrix.png')


if __name__ == '__main__':
    main()
