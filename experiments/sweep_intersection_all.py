"""Remaining intersection sweeps:
  · v_d sweep (all 4 flocks same speed, 3-25 m/s)
  · intersection box size sweep
  · d_c (cooperation distance) sweep with R+90°
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.control_alpha import control_alpha
from flocking_lib.control_gamma import control_gamma
from flocking_lib.control_tau import control_tau, R_PLUS_90
from flocking_lib.control_beta_cross import control_beta_cross


def make_cross_flock_v(direction, N, params, geom, dist_back, lane_offset, v_d):
    """make_cross_flock with parameterizable v_d."""
    s = params['d_a']
    if direction == 'E':
        vel = np.array([+v_d, 0.0]); x0 = -dist_back; y0 = +lane_offset; ax = 'x'
    elif direction == 'W':
        vel = np.array([-v_d, 0.0]); x0 = +dist_back; y0 = -lane_offset; ax = 'x'
    elif direction == 'N':
        vel = np.array([0.0, +v_d]); x0 = +lane_offset; y0 = -dist_back; ax = 'y'
    else:
        vel = np.array([0.0, -v_d]); x0 = -lane_offset; y0 = +dist_back; ax = 'y'
    q = np.zeros((N, 2)); p = np.tile(vel, (N, 1))
    for k in range(N):
        if ax == 'x':
            sign = +1 if direction == 'E' else -1
            q[k, 0] = x0 - sign * k * s; q[k, 1] = y0
        else:
            sign = +1 if direction == 'N' else -1
            q[k, 0] = x0; q[k, 1] = y0 - sign * k * s
    return q, p


def run_cross_v(v_d, base, geom, T=22.0, TS=0.02, c2_t=0.15, d_c=70.0,
                 N=4, dist_back=60.0, lane_offset=3.5, a_max=9.0):
    params = dict(base)
    params['tau_matrix'] = R_PLUS_90
    params['c1_t'] = 0.0; params['c2_t'] = c2_t; params['d_c'] = d_c
    params.pop('predict_suppress_threshold', None)
    flock_inits = []; p_d_dict = {}
    for k, d in enumerate(['E', 'W', 'N', 'S'], start=1):
        q, p = make_cross_flock_v(d, N, params, geom, dist_back, lane_offset, v_d)
        flock_inits.append((q, p)); p_d_dict[k] = p[0].copy()
    params['p_d_per_flock'] = p_d_dict
    num_steps = int(round(T / TS)) + 1
    N_total = sum(q.shape[0] for q, _ in flock_inits)
    q_arr = np.zeros((N_total, 2, num_steps))
    p_arr = np.zeros((N_total, 2, num_steps))
    flock_id = np.zeros(N_total, dtype=int)
    offset = 0
    for k, (qi, pi) in enumerate(flock_inits, start=1):
        n = qi.shape[0]
        q_arr[offset:offset+n, :, 0] = qi; p_arr[offset:offset+n, :, 0] = pi
        flock_id[offset:offset+n] = k; offset += n
    for t in range(num_steps - 1):
        qt, pt = q_arr[:,:,t], p_arr[:,:,t]
        ut = np.zeros((N_total, 2))
        for i in range(N_total):
            ut[i] = (control_alpha(i, qt, pt, flock_id, params)
                     + control_beta_cross(i, qt, pt, geom, params)
                     + control_gamma(i, qt, pt, flock_id, params)
                     + control_tau  (i, qt, pt, flock_id, params))
        for i in range(N_total):
            mag = np.linalg.norm(ut[i])
            if mag > a_max:
                ut[i] = a_max * ut[i] / mag
        p_arr[:,:,t+1] = pt + TS * ut
        q_arr[:,:,t+1] = qt + TS * p_arr[:,:,t+1]
    return q_arr, p_arr, flock_id


def metrics(q, flock_id, geom):
    N = q.shape[0]; inter = np.inf
    for ti in range(q.shape[2]):
        for i in range(N):
            for j in range(i+1, N):
                if flock_id[i] == flock_id[j]: continue
                d = float(np.linalg.norm(q[i,:,ti] - q[j,:,ti]))
                if d < inter: inter = d
    off = sum(1 for ti in range(q.shape[2]) for i in range(N)
              if abs(q[i,0,ti]) > geom['half_road']
              and abs(q[i,1,ti]) > geom['half_road'])
    return inter, off


def main():
    base = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,
    }
    geom_def = {'half_road': 7.0, 'half_inter': 7.0}

    # 1. v_d sweep
    print('=== INTERSECTION v_d sweep (R+90°, 4×4 flocks, c2_t=0.15, d_c=70) ===')
    print(f'{"v_d":>6}  {"inter":>7}  {"off-road":>9}')
    vd_rows = []
    for v_d in [3, 5, 7, 10, 15, 20, 25]:
        T = max(15.0, 8.0 + 120.0 / v_d)
        q, p, fid = run_cross_v(v_d, base, geom_def, T=T)
        inter, off = metrics(q, fid, geom_def)
        vd_rows.append((v_d, inter, off))
        print(f'{v_d:6.1f}  {inter:6.2f}m  {off:8d}')

    # 2. Road width sweep (vary half_road, half_inter = half_road for clean cross)
    print('\n=== INTERSECTION road-width sweep (half_road = half_inter) ===')
    print(f'{"hr":>6}  {"inter":>7}  {"off-road":>9}')
    rw_rows = []
    for hr in [5, 6, 7, 9, 12, 16]:
        geom = {'half_road': hr, 'half_inter': hr}
        lane_off = hr / 2
        q, p, fid = run_cross_v(10.0, base, geom, T=22.0, lane_offset=lane_off)
        inter, off = metrics(q, fid, geom)
        rw_rows.append((hr, inter, off))
        print(f'{hr:6.1f}  {inter:6.2f}m  {off:8d}')

    # 3. d_c sweep
    print('\n=== INTERSECTION d_c sweep (R+90°, v_d=10) ===')
    print(f'{"d_c":>6}  {"inter":>7}  {"off-road":>9}')
    dc_rows = []
    for d_c in [30, 50, 70, 90, 120, 150]:
        q, p, fid = run_cross_v(10.0, base, geom_def, d_c=d_c, T=22.0)
        inter, off = metrics(q, fid, geom_def)
        dc_rows.append((d_c, inter, off))
        print(f'{d_c:6.1f}  {inter:6.2f}m  {off:8d}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, rows, xlabel, title in [
        (axes[0], vd_rows, 'v_d [m/s]', 'v_d sweep'),
        (axes[1], rw_rows, 'half_road [m]', 'road-width sweep'),
        (axes[2], dc_rows, 'd_c [m]', 'd_c sweep'),
    ]:
        xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
        offs = [r[2] for r in rows]
        ax.plot(xs, ys, '-o', color='C0', label='inter-flock min')
        ax.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='car width 2 m')
        ax.set_xlabel(xlabel); ax.set_ylabel('inter [m]')
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.bar(xs, offs, alpha=0.3, color='crimson', width=(xs[-1]-xs[0])/len(xs)*0.5)
        ax2.set_ylabel('off-road', color='crimson')
    fig.suptitle('Intersection sweeps (V2 = R+90°, 4×4 flocks, walls)')
    fig.tight_layout()
    fig.savefig('sweep_intersection_all.png', dpi=100, bbox_inches='tight')
    print('\nsaved sweep_intersection_all.png')


if __name__ == '__main__':
    main()
