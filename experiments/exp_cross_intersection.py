"""Investigation: 4-way cross intersection with hard road walls.

Two perpendicular roads of width 14 m meeting at a 14×14 intersection box.
4 flocks, one approaching from each cardinal direction in its right-hand lane:
   * eastbound  (+x): northern lane of horizontal road, y =  +half_lane
   * westbound  (-x): southern lane, y = -half_lane
   * northbound (+y): eastern lane of vertical road, x = +half_lane
   * southbound (-y): western lane, x = -half_lane

We test V0 (J matrix, McKenzie baseline) vs V2 (R+90°, tuned).
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from flocking_lib.control_alpha import control_alpha
from flocking_lib.control_gamma import control_gamma
from flocking_lib.control_tau import control_tau, J, R_PLUS_90
from flocking_lib.control_beta_cross import control_beta_cross


def make_cross_flock(direction, N, params, geom, dist_back, lane_offset):
    """Build a single-row flock in its right-hand lane heading toward origin.
    direction: 'E', 'W', 'N', 'S'.
    """
    s = params['d_a']; v_d = 10.0
    if direction == 'E':                              # heading +x
        vel = np.array([+v_d, 0.0])
        x0 = -dist_back; y0 = +lane_offset
        axis_along = 'x'
    elif direction == 'W':                            # heading -x
        vel = np.array([-v_d, 0.0])
        x0 = +dist_back; y0 = -lane_offset
        axis_along = 'x'
    elif direction == 'N':                            # heading +y
        vel = np.array([0.0, +v_d])
        x0 = +lane_offset; y0 = -dist_back
        axis_along = 'y'
    elif direction == 'S':                            # heading -y
        vel = np.array([0.0, -v_d])
        x0 = -lane_offset; y0 = +dist_back
        axis_along = 'y'
    else:
        raise ValueError(direction)
    q = np.zeros((N, 2)); p = np.tile(vel, (N, 1))
    for k in range(N):
        if axis_along == 'x':
            # train extends backward along motion direction
            sign = +1 if direction == 'E' else -1
            q[k, 0] = x0 - sign * k * s
            q[k, 1] = y0
        else:
            sign = +1 if direction == 'N' else -1
            q[k, 0] = x0
            q[k, 1] = y0 - sign * k * s
    return q, p


def run_cross_sim(tau_matrix, c2_t, d_c, params_base, geom, T=18.0, TS=0.02,
                  N_per_flock=4, dist_back=60.0, lane_offset=3.5,
                  a_max=9.0, c1_t=0.0):
    """Run 4-flock cross intersection."""
    params = dict(params_base)
    params['tau_matrix'] = tau_matrix
    params['c2_t'] = c2_t
    params['c1_t'] = c1_t
    params['d_c']  = d_c
    # do NOT use predictive suppression (breaks roundabout)
    params.pop('predict_suppress_threshold', None)

    flock_inits = []
    p_d_dict = {}
    for k, d in enumerate(['E', 'W', 'N', 'S'], start=1):
        q, p = make_cross_flock(d, N_per_flock, params, geom, dist_back, lane_offset)
        flock_inits.append((q, p))
        p_d_dict[k] = p[0].copy()
    params['p_d_per_flock'] = p_d_dict

    num_steps = int(round(T / TS)) + 1
    N_total = sum(q.shape[0] for q, _ in flock_inits)
    q_arr = np.zeros((N_total, 2, num_steps))
    p_arr = np.zeros((N_total, 2, num_steps))
    flock_id = np.zeros(N_total, dtype=int)
    offset = 0
    for k, (qi, pi) in enumerate(flock_inits, start=1):
        n = qi.shape[0]
        q_arr[offset:offset+n, :, 0] = qi
        p_arr[offset:offset+n, :, 0] = pi
        flock_id[offset:offset+n] = k
        offset += n

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


def draw_road_geometry(ax, geom, extent):
    """Draw the "+" road walls."""
    hr = geom['half_road']; hi = geom['half_inter']
    # horizontal arm walls: y = ±half_road for |x| > half_inter
    ax.plot([-extent, -hi], [-hr, -hr], 'k-', linewidth=2)
    ax.plot([+hi, +extent], [-hr, -hr], 'k-', linewidth=2)
    ax.plot([-extent, -hi], [+hr, +hr], 'k-', linewidth=2)
    ax.plot([+hi, +extent], [+hr, +hr], 'k-', linewidth=2)
    # vertical arm walls: x = ±half_road for |y| > half_inter
    ax.plot([-hr, -hr], [-extent, -hi], 'k-', linewidth=2)
    ax.plot([-hr, -hr], [+hi, +extent], 'k-', linewidth=2)
    ax.plot([+hr, +hr], [-extent, -hi], 'k-', linewidth=2)
    ax.plot([+hr, +hr], [+hi, +extent], 'k-', linewidth=2)
    # lane divider (dashed at centerlines)
    ax.plot([-extent, -hi], [0, 0], 'k--', linewidth=0.5, alpha=0.4)
    ax.plot([+hi, +extent], [0, 0], 'k--', linewidth=0.5, alpha=0.4)
    ax.plot([0, 0], [-extent, -hi], 'k--', linewidth=0.5, alpha=0.4)
    ax.plot([0, 0], [+hi, +extent], 'k--', linewidth=0.5, alpha=0.4)


def main():
    base = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,
    }
    geom = {'half_road': 7.0, 'half_inter': 7.0}

    q, p, fid = run_cross_sim(R_PLUS_90, c2_t=0.15, d_c=70.0,
                              params_base=base, geom=geom, T=18.0,
                              dist_back=60.0, lane_offset=3.5)

    # ---- Snapshot strip --------------------------------------------------
    TS = 0.02
    strip_times = np.linspace(0, q.shape[2]-1, 8).astype(int)
    flock_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    extent = max(abs(q[:,0,:]).max(), abs(q[:,1,:]).max()) + 5
    for k, t in enumerate(strip_times):
        ax = axes[k // 4, k % 4]
        draw_road_geometry(ax, geom, extent)
        for i in range(q.shape[0]):
            c = flock_colors[(fid[i]-1) % 4]
            ax.scatter(q[i,0,t], q[i,1,t], c=c, s=60, edgecolors='k', linewidth=0.5)
            vx, vy = p[i,0,t], p[i,1,t]
            ax.arrow(q[i,0,t], q[i,1,t], vx*0.5, vy*0.5,
                     head_width=2, color=c, alpha=0.5)
        ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.set_title(f't = {t*TS:.2f} s', fontsize=10)
    fig.suptitle('4-way cross intersection with hard road walls — V2 (R+90°)')
    fig.tight_layout()
    fig.savefig('cross_intersection_snapshots.png', dpi=110, bbox_inches='tight')

    # ---- GIF -------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    draw_road_geometry(ax2, geom, extent)
    ax2.set_xlim(-extent, extent); ax2.set_ylim(-extent, extent)
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.2)
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]')
    cols = [flock_colors[(fid[i]-1) % 4] for i in range(q.shape[0])]
    scat = ax2.scatter(q[:,0,0], q[:,1,0], c=cols, s=70, edgecolors='k')
    title = ax2.set_title('4-way cross intersection — t = 0.00 s')
    def update(frame):
        scat.set_offsets(q[:,:,frame])
        title.set_text(f'4-way cross intersection — t = {frame*TS:.2f} s')
        return scat, title
    anim = manimation.FuncAnimation(fig2, update,
                                     frames=range(0, q.shape[2], 5),
                                     interval=80, blit=False)
    anim.save('cross_intersection.gif', writer='pillow', fps=15, dpi=70)

    # Inter-flock min
    inter = np.inf; t_min = 0
    for ti in range(q.shape[2]):
        for i in range(q.shape[0]):
            for j in range(i+1, q.shape[0]):
                if fid[i] == fid[j]: continue
                d = float(np.linalg.norm(q[i,:,ti] - q[j,:,ti]))
                if d < inter: inter, t_min = d, ti
    # Track escapes (cars in the off-road corner zone)
    off_road = 0
    for ti in range(q.shape[2]):
        for i in range(q.shape[0]):
            x, y = q[i,0,ti], q[i,1,ti]
            if abs(x) > geom['half_road'] and abs(y) > geom['half_road']:
                off_road += 1
    print(f'inter-flock min: {inter:.2f} m at t={t_min*TS:.2f}s')
    print(f'off-road (agent·step) count: {off_road}')


if __name__ == '__main__':
    main()
