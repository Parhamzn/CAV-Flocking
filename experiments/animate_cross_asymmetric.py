"""Render GIFs for each asymmetric cross-intersection scenario."""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from exp_cross_asymmetric import run_cross_asym, metrics
from exp_cross_intersection import draw_road_geometry


def render_gif(label, sizes, base, geom, T=20.0, TS=0.02, outname='cross.gif'):
    q, p, fid, labels = run_cross_asym(sizes, base, geom, T=T, TS=TS)
    inter, off_road, finals = metrics(q, p, fid, geom)
    flock_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']

    fig, ax = plt.subplots(figsize=(8, 8))
    ext = max(80, max(abs(q[:,0,:]).max(), abs(q[:,1,:]).max()) + 5)
    draw_road_geometry(ax, geom, ext)
    ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    cols = [flock_colors[(fid[i]-1) % len(flock_colors)] for i in range(q.shape[0])]
    scat = ax.scatter(q[:,0,0], q[:,1,0], c=cols, s=60, edgecolors='k', linewidth=0.5)
    title = ax.set_title(f'{label}\nt = 0.00 s | inter-min so far = ∞', fontsize=10)

    # Track running inter-min for the title
    N = q.shape[0]
    running_inter = []
    cur = np.inf
    for ti in range(q.shape[2]):
        for i in range(N):
            for j in range(i+1, N):
                if fid[i] == fid[j]: continue
                d = float(np.linalg.norm(q[i,:,ti] - q[j,:,ti]))
                if d < cur: cur = d
        running_inter.append(cur)

    def update(frame):
        scat.set_offsets(q[:,:,frame])
        rm = running_inter[frame]
        rm_str = '∞' if not np.isfinite(rm) else f'{rm:.2f}m'
        title.set_text(f'{label}\nt = {frame*TS:.2f} s | inter-min so far = {rm_str}')
        return scat, title

    anim = manimation.FuncAnimation(fig, update,
                                     frames=range(0, q.shape[2], 5),
                                     interval=80, blit=False)
    anim.save(outname, writer='pillow', fps=15, dpi=70)
    plt.close(fig)
    return inter, off_road, finals


def main():
    base = {
        'e':0.1,'a':5,'b':5,'d_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,
    }
    geom = {'half_road': 7.0, 'half_inter': 7.0}

    scenarios = [
        ('A · baseline 4-4-4-4',         {'E':4,'W':4,'N':4,'S':4}, 'cross_A_baseline.gif'),
        ('B · big eastbound 8-2-2-2',    {'E':8,'W':2,'N':2,'S':2}, 'cross_B_bigE.gif'),
        ('C · big horizontal 8-8-2-2',   {'E':8,'W':8,'N':2,'S':2}, 'cross_C_bigHorz.gif'),
        ('D · 3-way (no southbound)',    {'E':4,'W':4,'N':4,'S':0}, 'cross_D_3way.gif'),
        ('E · 2-way opposing (E+W)',     {'E':4,'W':4,'N':0,'S':0}, 'cross_E_2opposing.gif'),
        ('F · 2-way perpendicular (E+N)',{'E':4,'W':0,'N':4,'S':0}, 'cross_F_2perp.gif'),
        ('H · diagonal mismatch (E+S)',  {'E':4,'W':0,'N':0,'S':4}, 'cross_H_diag.gif'),
    ]

    for label, sizes, outname in scenarios:
        inter, off_road, finals = render_gif(label, sizes, base, geom, outname=outname)
        print(f'{outname:30s}  inter={inter:.2f}m off-road={off_road}')


if __name__ == '__main__':
    main()
