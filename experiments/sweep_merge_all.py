"""Remaining merge sweeps:
  · L_merge (merge zone length) sweep
  · Speed mismatch (main v_d=10, on-ramp v_d varied)
  · Density (N_ramp sweep at fixed N_main)
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.flock_layouts import grid_formation
from flocking_lib.control_alpha import control_alpha
from flocking_lib.control_gamma import control_gamma
from flocking_lib.control_tau import control_tau
from flocking_lib.control_beta_merge import control_beta_merge


def run_merge_speedmismatch(N_main, N_ramp, v_main, v_ramp, base, geom,
                             ramp_stagger=0.0, T=18.0, TS=0.02, a_max=9.0,
                             lead_x_main=-40.0):
    s = base['d_a']
    flock_inits = []; p_d_dict = {}
    if N_main > 0:
        v = np.array([+v_main, 0.0])
        xc = lead_x_main - (N_main - 1) * s / 2 - (N_main - 1) * s + (N_main - 1) * s
        xc = lead_x_main - (N_main - 1) * s / 2
        q1, p1 = grid_formation(1, N_main, x_center=xc, y_center=3.5,
                                spacing=s, vel=v)
        flock_inits.append((q1, p1)); p_d_dict[1] = v.copy()
    if N_ramp > 0:
        v = np.array([+v_ramp, 0.0])
        xc = lead_x_main - ramp_stagger - (N_ramp - 1) * s / 2
        q2, p2 = grid_formation(1, N_ramp, x_center=xc, y_center=-3.5,
                                spacing=s, vel=v)
        flock_inits.append((q2, p2)); p_d_dict[2 if N_main > 0 else 1] = v.copy()

    num_steps = int(round(T / TS)) + 1
    N_total = sum(q.shape[0] for q, _ in flock_inits)
    q = np.zeros((N_total, 2, num_steps))
    p = np.zeros((N_total, 2, num_steps))
    flock_id = np.ones(N_total, dtype=int)             # single-flock approach
    offset = 0
    for (qi, pi) in flock_inits:
        n = qi.shape[0]
        q[offset:offset+n, :, 0] = qi
        p[offset:offset+n, :, 0] = pi
        offset += n
    params = dict(base, p_d_per_flock={1: np.array([+10.0, 0.0])})
    # For γ, use main road's desired velocity (handles both flocks consistently)

    for t in range(num_steps - 1):
        qt, pt = q[:,:,t], p[:,:,t]
        ut = np.zeros((N_total, 2))
        for i in range(N_total):
            ut[i] = (control_alpha(i, qt, pt, flock_id, params)
                     + control_beta_merge(i, qt, pt, geom, params)
                     + control_gamma(i, qt, pt, flock_id, params)
                     + control_tau  (i, qt, pt, flock_id, params))
        for i in range(N_total):
            mag = np.linalg.norm(ut[i])
            if mag > a_max:
                ut[i] = a_max * ut[i] / mag
        p[:,:,t+1] = pt + TS * ut
        q[:,:,t+1] = qt + TS * p[:,:,t+1]
    return q, p, flock_id


def metrics(q, geom):
    N = q.shape[0]; pmin = np.inf
    for ti in range(q.shape[2]):
        for i in range(N):
            for j in range(i+1, N):
                d = float(np.linalg.norm(q[i,:,ti] - q[j,:,ti]))
                if d < pmin: pmin = d
    L = geom['L_merge']; mt = geom['main_top']; mb = geom['main_bot']; rb = geom['ramp_bot']
    off = 0
    for ti in range(q.shape[2]):
        for i in range(N):
            x, y = q[i,0,ti], q[i,1,ti]
            if x < 0:
                in_road = (mb <= y <= mt) or (rb <= y <= mb)
            elif x <= L:
                bot = rb + (mb - rb) * (x / L)
                in_road = (bot - 0.1 <= y <= mt + 0.1)
            else:
                in_road = (mb - 0.1 <= y <= mt + 0.1)
            if not in_road: off += 1
    in_main = sum(1 for i in range(N) if q[i, 1, -1] > geom['main_bot'])
    return pmin, off, in_main


def main():
    base = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,'d_c':40.0,'c1_t':0.0,'c2_t':0.08,
    }
    geom_def = {'L_merge': 30.0, 'main_top': 7.0, 'main_bot': 0.0, 'ramp_bot': -7.0}

    # 1. L_merge sweep
    print('=== MERGE L_merge sweep (4 main + 4 ramp, no stagger, v_d=10) ===')
    print(f'{"L":>6}  {"pair":>7}  {"off-road":>9}  {"in_main":>9}')
    Lm_rows = []
    for L in [5, 10, 20, 30, 50, 80]:
        geom = dict(geom_def, L_merge=L)
        q, p, fid = run_merge_speedmismatch(4, 4, 10.0, 10.0, base, geom, T=18.0)
        pmin, off, in_main = metrics(q, geom)
        Lm_rows.append((L, pmin, off, in_main))
        print(f'{L:6.1f}  {pmin:6.2f}m  {off:8d}  {in_main:9d}/8')

    # 2. Speed mismatch sweep (main fixed at 10, ramp varies)
    print('\n=== MERGE speed mismatch (main v=10, ramp v varies) ===')
    print(f'{"v_ramp":>7}  {"pair":>7}  {"off-road":>9}  {"in_main":>9}')
    vm_rows = []
    for v_ramp in [3, 5, 7, 10, 13, 15, 18]:
        # extend T so slow ramp still merges in
        T = max(18.0, 6.0 + 120.0 / min(10.0, v_ramp))
        q, p, fid = run_merge_speedmismatch(4, 4, 10.0, v_ramp, base, geom_def, T=T)
        pmin, off, in_main = metrics(q, geom_def)
        vm_rows.append((v_ramp, pmin, off, in_main))
        print(f'{v_ramp:7.1f}  {pmin:6.2f}m  {off:8d}  {in_main:9d}/8')

    # 3. Density sweep (N_ramp varies at N_main=4)
    print('\n=== MERGE density sweep (N_main=4, N_ramp varies) ===')
    print(f'{"N_ramp":>7}  {"pair":>7}  {"off-road":>9}  {"in_main":>9}')
    dens_rows = []
    for N_ramp in [1, 2, 3, 4, 6, 8, 10, 12]:
        q, p, fid = run_merge_speedmismatch(4, N_ramp, 10.0, 10.0, base, geom_def, T=20.0)
        pmin, off, in_main = metrics(q, geom_def)
        dens_rows.append((N_ramp, pmin, off, in_main))
        print(f'{N_ramp:7d}  {pmin:6.2f}m  {off:8d}  {in_main:9d}/{4+N_ramp}')

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, rows, xlabel, title in [
        (axes[0], Lm_rows,  'L_merge [m]',           'L_merge sweep'),
        (axes[1], vm_rows,  'on-ramp v_d [m/s]',     'speed mismatch'),
        (axes[2], dens_rows, 'N_ramp (N_main=4)',     'density sweep'),
    ]:
        xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
        offs = [r[2] for r in rows]
        ax.plot(xs, ys, '-o', color='C2', label='pair_min')
        ax.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='car width 2 m')
        ax.set_xlabel(xlabel); ax.set_ylabel('pair-min [m]')
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.bar(xs, offs, alpha=0.3, color='crimson',
                width=(max(xs)-min(xs))/len(xs)*0.5)
        ax2.set_ylabel('off-road', color='crimson')
    fig.suptitle('Merge sweeps (single-flock_id, v_main=10 unless noted)')
    fig.tight_layout()
    fig.savefig('sweep_merge_all.png', dpi=100, bbox_inches='tight')
    print('\nsaved sweep_merge_all.png')


if __name__ == '__main__':
    main()
