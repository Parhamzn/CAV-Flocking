"""Asymmetric merge configurations: vary N_main, N_ramp, and arrival stagger.

Using the "single flock_id" approach (α-lattice acts across both streams)
since the two-flock-id version produces complete overlap (no inter-flock
interaction for same-direction flows).
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from flocking_lib.flock_layouts import grid_formation
from flocking_lib.control_alpha import control_alpha
from flocking_lib.control_gamma import control_gamma
from flocking_lib.control_tau import control_tau
from flocking_lib.control_beta_merge import control_beta_merge
from exp_merge import draw_merge_geometry


def run_asym_merge(N_main, N_ramp, base_params, geom,
                   ramp_stagger=0.0, T=18.0, TS=0.02, a_max=9.0,
                   v_d=10.0, lead_x_main=-50.0):
    """Run a merge with given main / ramp sizes and stagger.

    ramp_stagger: how far behind the main flock the on-ramp flock starts
        (positive = on-ramp arrives at merge zone later than main).
    """
    s = base_params['d_a']
    vel = np.array([+v_d, 0.0])

    flock_inits = []
    if N_main > 0:
        # Place flock so its leading car is at lead_x_main
        x_center = lead_x_main - (N_main - 1) * s / 2
        q_main, p_main = grid_formation(1, N_main, x_center=x_center,
                                         y_center=3.5, spacing=s, vel=vel)
        flock_inits.append((q_main, p_main))
    if N_ramp > 0:
        x_center = lead_x_main - ramp_stagger - (N_ramp - 1) * s / 2
        q_ramp, p_ramp = grid_formation(1, N_ramp, x_center=x_center,
                                         y_center=-3.5, spacing=s, vel=vel)
        flock_inits.append((q_ramp, p_ramp))

    num_steps = int(round(T / TS)) + 1
    N_total = sum(q.shape[0] for q, _ in flock_inits)
    q = np.zeros((N_total, 2, num_steps))
    p = np.zeros((N_total, 2, num_steps))
    flock_id = np.ones(N_total, dtype=int)            # single-flock approach
    offset = 0
    for (qi, pi) in flock_inits:
        n = qi.shape[0]
        q[offset:offset+n, :, 0] = qi
        p[offset:offset+n, :, 0] = pi
        offset += n

    params = dict(base_params, p_d_per_flock={1: vel.copy()})

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


def metrics(q, geom, N_main, N_ramp, TS):
    N = q.shape[0]
    pair_min = np.inf
    for ti in range(q.shape[2]):
        for i in range(N):
            for j in range(i+1, N):
                d = float(np.linalg.norm(q[i,:,ti] - q[j,:,ti]))
                if d < pair_min: pair_min = d
    # off-road
    L = geom['L_merge']
    main_top = geom['main_top']; main_bot = geom['main_bot']; ramp_bot = geom['ramp_bot']
    off = 0
    for ti in range(q.shape[2]):
        for i in range(N):
            x, y = q[i,0,ti], q[i,1,ti]
            if x < 0:
                in_road = (main_bot <= y <= main_top) or (ramp_bot <= y <= main_bot)
            elif x <= L:
                bot = ramp_bot + (main_bot - ramp_bot) * (x / L)
                in_road = (bot - 0.1 <= y <= main_top + 0.1)
            else:
                in_road = (main_bot - 0.1 <= y <= main_top + 0.1)
            if not in_road:
                off += 1
    # Did the on-ramp cars make it into the main road?
    # By the end of the run, all cars should have y > main_bot.
    final_in_main = sum(1 for i in range(N) if q[i, 1, -1] > main_bot)
    # Stall check: any car with terminal speed < 0.5 v_d (= 5 m/s)
    final_speeds = np.linalg.norm(p_arr_dummy_call(q, TS), axis=1) \
        if False else np.linalg.norm((q[:, :, -1] - q[:, :, -10]) / (9*TS), axis=1)
    stalled = int(np.sum(final_speeds < 5.0))
    return pair_min, off, final_in_main, stalled


def p_arr_dummy_call(q, TS):
    # Just for signature compatibility
    return (q[:, :, 1:] - q[:, :, :-1]) / TS


def main():
    base = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,'d_c':40.0,'c1_t':0.0,'c2_t':0.08,
    }
    geom = {'L_merge': 30.0, 'main_top': 7.0, 'main_bot': 0.0, 'ramp_bot': -7.0}

    scenarios = [
        ('A · baseline 4+4',           4, 4, 0.0),
        ('B · big main 8+2',           8, 2, 0.0),
        ('C · big ramp 2+8',           2, 8, 0.0),
        ('D · empty main 0+4',         0, 4, 0.0),
        ('E · empty ramp 4+0',         4, 0, 0.0),
        ('F · stagger ramp +30m',      4, 4, 30.0),
        ('G · single car merge 4+1',   4, 1, 0.0),
        ('H · long main 10+2',        10, 2, 0.0),
    ]

    TS = 0.02
    print(f'{"scenario":<32}  {"pair":>6}  {"off-road":>9}  {"in-main/total":>14}  {"stalled":>7}')
    print('-' * 80)
    results = []
    for label, n_main, n_ramp, stagger in scenarios:
        q, p, fid = run_asym_merge(n_main, n_ramp, base, geom,
                                    ramp_stagger=stagger, T=18.0, TS=TS,
                                    lead_x_main=-40.0)
        pmin, off, in_main, stalled = metrics(q, geom, n_main, n_ramp, TS)
        N_total = n_main + n_ramp
        results.append((label, n_main, n_ramp, stagger, q, p, fid, pmin, off, in_main, stalled))
        print(f'{label:<32}  {pmin:5.2f}m  {off:8d}  {in_main:5d}/{N_total:<8d}  {stalled:7d}')

    # Render snapshot grids: 4 time slices per scenario, stacked vertically
    fig, axes = plt.subplots(len(results), 4, figsize=(13, 2 * len(results)))
    for row, (label, n_main, n_ramp, stagger, q, p, fid, pmin, off, in_main, stalled) in enumerate(results):
        strip_times = np.linspace(0, q.shape[2]-1, 4).astype(int)
        if q.shape[0] == 0:
            for col in range(4):
                axes[row, col].set_visible(False)
            continue
        x_min = q[:,0,:].min() - 5; x_max = q[:,0,:].max() + 5
        for col, t in enumerate(strip_times):
            ax = axes[row, col] if len(results) > 1 else axes[col]
            draw_merge_geometry(ax, geom, x_min, x_max)
            for i in range(q.shape[0]):
                # color by original lane
                if i < n_main:
                    c = 'tab:blue'
                else:
                    c = 'tab:red'
                ax.scatter(q[i,0,t], q[i,1,t], c=c, s=30, edgecolors='k', linewidth=0.3)
            ax.set_xlim(x_min, x_max); ax.set_ylim(-12, 12)
            ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
            if col == 0: ax.set_ylabel(label, fontsize=7, weight='bold')
            ax.set_title(f't={t*TS:.1f}s', fontsize=8)
    fig.suptitle('Asymmetric merge configurations', fontsize=12)
    fig.tight_layout()
    fig.savefig('merge_asymmetric.png', dpi=70, bbox_inches='tight')
    plt.close(fig)
    print('\nsaved merge_asymmetric.png')


if __name__ == '__main__':
    main()
