"""Tau-off baseline: reproduce McKenzie's "Without Tau-agent" comparison.

In my implementation alpha-interactions are restricted to same-flock pairs;
the tau-agent is the only inter-flock coordination. So "turning off tau" means
the two opposing flocks have NO interaction at all — they pass through each
other as ghosts (point-mass dynamics, no collision response).

Compare with-tau (corrected algorithm: c1_t=0, c2_t=0.08, d_c=40) vs no-tau
(c1_t=c2_t=0) across flock sizes, reporting inter-flock min and the "ghost
overlap" count: number of (i, j, t) triples where two opposing-flock agents
are within 1 m of each other.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from sweep_n_per_flock import run_sim


def metrics_full(q, flock_id, y_hi, d_b):
    N = q.shape[0]; N1 = (flock_id == 1).sum()
    inter_min = np.inf
    overlap_count = 0
    for t in range(q.shape[2]):
        for i in range(N1):
            for j in range(N1, N):
                d = np.linalg.norm(q[i, :, t] - q[j, :, t])
                if d < inter_min:
                    inter_min = d
                if d < 1.0:        # arbitrary "collision proxy" threshold
                    overlap_count += 1
    near_wall = ((q[:, 1, :] < d_b) | (q[:, 1, :] > y_hi - d_b))
    return dict(inter_min=inter_min, overlap_count=overlap_count,
                wall_proximity_fraction=near_wall.mean())


def main():
    base_params = {
        'e':0.1,'a':5,'b':5,
        'd_a':7,'r_a':1.2*7,'h_a':0.2,'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5, 'd_c': 40.0,
        'p_d_flock1': np.array([+10.0, 0.0]),
        'p_d_flock2': np.array([-10.0, 0.0]),
    }
    with_tau = dict(base_params, c1_t=0.0,  c2_t=0.08)
    no_tau   = dict(base_params, c1_t=0.0,  c2_t=0.0)
    scenario = dict(TS=0.02, T=14.0, a_max=9.0, x_inner=20.0, y_hi=14.4)

    Ns = [2, 4, 6, 8, 10]
    rows = []
    for N_pf in Ns:
        q_on, _, fid_on   = run_sim(N_pf, with_tau, scenario)
        q_off, _, fid_off = run_sim(N_pf, no_tau,   scenario)
        m_on  = metrics_full(q_on,  fid_on,  scenario['y_hi'], with_tau['d_b'])
        m_off = metrics_full(q_off, fid_off, scenario['y_hi'], no_tau['d_b'])
        rows.append((N_pf, m_on, m_off))
        print(f'N={N_pf:2d}  WITH τ: inter={m_on["inter_min"]:5.2f}m overlap={m_on["overlap_count"]:5d}'
              f'   ||   NO τ: inter={m_off["inter_min"]:5.2f}m overlap={m_off["overlap_count"]:5d}')

    Ns_a = [r[0] for r in rows]
    inter_on  = [r[1]['inter_min'] for r in rows]
    inter_off = [r[2]['inter_min'] for r in rows]
    overlap_on  = [r[1]['overlap_count'] for r in rows]
    overlap_off = [r[2]['overlap_count'] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(Ns_a, inter_on,  '-o', label='with τ (corrected algorithm)')
    axes[0].plot(Ns_a, inter_off, '-s', label='no τ (α + β + γ only)')
    axes[0].axhline(2.0, color='r', linestyle='--', alpha=0.4, label='car width 2 m')
    axes[0].set_xlabel('cars per flock'); axes[0].set_ylabel('inter-flock min [m]')
    axes[0].set_title('Inter-flock minimum distance')
    axes[0].legend(fontsize=9); axes[0].grid(True)

    w = 0.35
    xs = np.arange(len(Ns_a))
    axes[1].bar(xs - w/2, overlap_on,  width=w, label='with τ', color='C0')
    axes[1].bar(xs + w/2, overlap_off, width=w, label='no τ',   color='C1')
    axes[1].set_xticks(xs); axes[1].set_xticklabels([str(n) for n in Ns_a])
    axes[1].set_xlabel('cars per flock'); axes[1].set_ylabel('# (i,j,t) triples with dist < 1 m')
    axes[1].set_title('Ghost-overlap count (collision proxy)')
    axes[1].legend(fontsize=9); axes[1].grid(True, axis='y')

    fig.suptitle('τ-off baseline — McKenzie Table 6.1 analog')
    fig.tight_layout()
    fig.savefig('sweep_tau_off.png', dpi=110, bbox_inches='tight')
    print('saved sweep_tau_off.png')


if __name__ == '__main__':
    main()
