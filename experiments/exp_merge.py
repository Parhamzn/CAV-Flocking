"""Investigate merging at a main-road / on-ramp junction.

Geometry: main road y in [0, 7] all x, on-ramp y in [-7, 0] for x<0,
merge zone 0 ≤ x ≤ 30 where the on-ramp's outer wall ramps up to meet
the main road's bottom. Both flocks travel +x at v_d=10 m/s.

Two configurations tested:
  · single-flock-id     — both lanes share one flock_id, so α-lattice
                          applies across them (cars naturally maintain
                          spacing as the on-ramp merges in)
  · two-flock-id        — separate IDs, so α only acts within-flock.
                          τ doesn't fire either (same headings, gate
                          closes). No inter-flock interaction at all —
                          shows the McKenzie baseline failure on merging.

Renders snapshot strip + GIF for each configuration.
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


def draw_merge_geometry(ax, geom, x_min, x_max):
    L = geom['L_merge']
    main_top = geom.get('main_top', 7.0)
    main_bot = geom.get('main_bot', 0.0)
    ramp_bot = geom.get('ramp_bot', -7.0)

    # Main road top wall (all x)
    ax.plot([x_min, x_max], [main_top, main_top], 'k-', linewidth=2)
    # Main road bottom (= gore for x<0, = ramping wall for x∈[0,L], = main bot for x>L)
    ax.plot([x_min, 0],     [main_bot, main_bot], 'k-', linewidth=2)    # gore divider for x<0
    ax.plot([L,    x_max],  [main_bot, main_bot], 'k-', linewidth=2)    # post-merge bottom
    # On-ramp outer (bottom) wall:
    ax.plot([x_min, 0],     [ramp_bot, ramp_bot], 'k-', linewidth=2)
    ax.plot([0,    L],      [ramp_bot, main_bot], 'k-', linewidth=2)    # ramping wall
    # Lane center dashed lines (cosmetic)
    ax.plot([x_min, x_max], [(main_top + main_bot)/2]*2, 'k--', linewidth=0.5, alpha=0.3)
    ax.plot([x_min, 0],     [(main_bot + ramp_bot)/2]*2, 'k--', linewidth=0.5, alpha=0.3)


def run_merge(use_single_flock_id, params, geom, T=14.0, TS=0.02,
              N_per_flock=4, a_max=9.0):
    s = params['d_a']; v_d = 10.0
    vel = np.array([+v_d, 0.0])
    # Main road flock: at y=3.5 (center of main road), x=[-110...-89]
    q_main, p_main = grid_formation(1, N_per_flock,
                                     x_center=-100 + (N_per_flock - 1)*s/2 - (N_per_flock - 1)*s,
                                     y_center=3.5, spacing=s, vel=vel)
    # On-ramp flock: at y=-3.5 (center of on-ramp), same x
    q_ramp, p_ramp = grid_formation(1, N_per_flock,
                                     x_center=-100 + (N_per_flock - 1)*s/2 - (N_per_flock - 1)*s,
                                     y_center=-3.5, spacing=s, vel=vel)

    flock_inits = [(q_main, p_main), (q_ramp, p_ramp)]
    p_d_dict = {1: vel.copy(), 2: vel.copy()}

    num_steps = int(round(T / TS)) + 1
    N_total = 2 * N_per_flock
    q = np.zeros((N_total, 2, num_steps))
    p = np.zeros((N_total, 2, num_steps))
    flock_id = np.zeros(N_total, dtype=int)
    if use_single_flock_id:
        flock_id[:] = 1
        p_d_dict = {1: vel.copy()}
    else:
        flock_id[:N_per_flock] = 1
        flock_id[N_per_flock:] = 2
    q[:N_per_flock, :, 0] = q_main
    q[N_per_flock:, :, 0] = q_ramp
    p[:N_per_flock, :, 0] = p_main
    p[N_per_flock:, :, 0] = p_ramp

    params = dict(params, p_d_per_flock=p_d_dict)

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


def metrics(q, flock_id, geom):
    N = q.shape[0]
    # min distance across all pairs (not just inter-flock)
    pair_min = np.inf
    inter_min = np.inf
    for ti in range(q.shape[2]):
        for i in range(N):
            for j in range(i+1, N):
                d = float(np.linalg.norm(q[i,:,ti] - q[j,:,ti]))
                if d < pair_min: pair_min = d
                if flock_id[i] != flock_id[j]:
                    if d < inter_min: inter_min = d
    # Off-road: car at (x,y) with y outside the road region for that x.
    L = geom['L_merge']
    main_top = geom.get('main_top', 7.0); main_bot = geom.get('main_bot', 0.0)
    ramp_bot = geom.get('ramp_bot', -7.0)
    off = 0
    for ti in range(q.shape[2]):
        for i in range(N):
            x, y = q[i,0,ti], q[i,1,ti]
            if x < 0:
                # either main (y in [main_bot, main_top]) or on-ramp (y in [ramp_bot, main_bot])
                in_road = (main_bot <= y <= main_top) or (ramp_bot <= y <= main_bot)
            elif x <= L:
                bot = ramp_bot + (main_bot - ramp_bot) * (x / L)
                in_road = (bot <= y <= main_top)
            else:
                in_road = (main_bot <= y <= main_top)
            if not in_road:
                off += 1
    return pair_min, inter_min, off


def render(label, q, p, flock_id, geom, outname_png, outname_gif):
    TS = 0.02
    colors = ['tab:blue', 'tab:red']
    cols = ['tab:blue' if fid == 1 else 'tab:red' for fid in flock_id]

    # Snapshot strip
    strip_times = np.linspace(0, q.shape[2]-1, 6).astype(int)
    fig, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=True)
    x_min, x_max = q[:,0,:].min() - 5, q[:,0,:].max() + 5
    for k, t in enumerate(strip_times):
        ax = axes[k]
        draw_merge_geometry(ax, geom, x_min, x_max)
        for i in range(q.shape[0]):
            ax.scatter(q[i,0,t], q[i,1,t], c=cols[i], s=40, edgecolors='k', linewidth=0.4)
            vx, vy = p[i,0,t], p[i,1,t]
            ax.arrow(q[i,0,t], q[i,1,t], vx*0.3, vy*0.3,
                     head_width=0.5, color=cols[i], alpha=0.5)
        ax.set_ylim(-12, 12)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.set_ylabel(f't={t*TS:.2f}s', fontsize=9)
    axes[-1].set_xlabel('x [m]')
    fig.suptitle(label, fontsize=11)
    fig.tight_layout()
    fig.savefig(outname_png, dpi=80, bbox_inches='tight')
    plt.close(fig)

    # GIF
    fig, ax = plt.subplots(figsize=(12, 4))
    draw_merge_geometry(ax, geom, x_min, x_max)
    ax.set_xlim(x_min, x_max); ax.set_ylim(-12, 12)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    scat = ax.scatter(q[:,0,0], q[:,1,0], c=cols, s=60, edgecolors='k', linewidth=0.5)
    title = ax.set_title(f'{label} — t = 0.00 s', fontsize=10)
    def update(frame):
        scat.set_offsets(q[:,:,frame])
        title.set_text(f'{label} — t = {frame*TS:.2f} s')
        return scat, title
    anim = manimation.FuncAnimation(fig, update,
                                     frames=range(0, q.shape[2], 5),
                                     interval=80, blit=False)
    anim.save(outname_gif, writer='pillow', fps=15, dpi=70)
    plt.close(fig)


def main():
    base = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,'d_c':40.0,'c1_t':0.0,'c2_t':0.08,
    }
    geom = {'L_merge': 30.0, 'main_top': 7.0, 'main_bot': 0.0, 'ramp_bot': -7.0}

    for use_single, label, png, gif in [
        (True,  'Single flock_id (α applies across both lanes)',
         'merge_single.png', 'merge_single.gif'),
        (False, 'Two flock_ids (α only within each lane, no τ fires)',
         'merge_two.png',    'merge_two.gif'),
    ]:
        q, p, fid = run_merge(use_single, base, geom, T=16.0)
        pair_min, inter_min, off = metrics(q, fid, geom)
        print(f'{label:<55}  pair_min={pair_min:.2f}m  inter_min={inter_min:.2f}m  off-road={off}')
        render(label, q, p, fid, geom, png, gif)


if __name__ == '__main__':
    main()
