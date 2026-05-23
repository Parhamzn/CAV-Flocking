"""Multi-seed run for Exp D to assess result robustness.

Run each λ with N_SEEDS independent seeds and aggregate intra-min and
throughput as (mean, p25, p75). The single-seed Exp D output is noisy
because intersection dynamics are quasi-chaotic, so a single bad seed
can produce a non-monotonic curve. We want to know if the capacity
threshold is a real algorithmic limit or seed luck.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import exp_intersection_mfd as M

N_SEEDS = 6
T_RUN = 40.0
SAFETY = M.D_A / 2


def sweep(lambdas, c2_t, d_c, label):
    M.C2_T = c2_t
    M.D_C = d_c
    print(f'\n=== {label}  (c2_t={c2_t}, d_c={d_c})  T={T_RUN}s  '
          f'seeds={N_SEEDS} ===')
    print(f'{"λ":>4}  {"q mean":>7}  {"q p25-p75":>10}  '
          f'{"intra med":>10}  {"intra p25":>10}  '
          f'{"safe%":>6}')
    out = []
    for lam in lambdas:
        runs = []
        for s in range(N_SEEDS):
            r = M.run_intersection_mfd(lam, T=T_RUN, seed=42 + s)
            runs.append(r)
        qs = np.array([r['throughput_total'] for r in runs])
        intras = np.array([r['min_pair_dist'] for r in runs])
        drops = np.array([100 * r['n_dropped'] /
                          max(1, r['n_dropped'] + r['n_injected']) for r in runs])
        safe_pct = 100.0 * (intras > SAFETY).mean()
        row = dict(lam=lam, q_mean=qs.mean(),
                   q_p25=np.percentile(qs, 25), q_p75=np.percentile(qs, 75),
                   intra_med=np.median(intras),
                   intra_p25=np.percentile(intras, 25),
                   intra_p75=np.percentile(intras, 75),
                   drop_mean=drops.mean(),
                   safe_pct=safe_pct)
        out.append(row)
        print(f'{lam:4.2f}  {qs.mean():7.2f}  '
              f'{np.percentile(qs, 25):4.2f}–{np.percentile(qs, 75):4.2f}  '
              f'{np.median(intras):10.2f}  {np.percentile(intras, 25):10.2f}  '
              f'{safe_pct:5.0f}%')
    return out


def main():
    lambdas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

    base = sweep(lambdas, c2_t=0.15, d_c=70, label='baseline V2 τ')
    tuned = sweep(lambdas, c2_t=0.20, d_c=100, label='λ=0.6 best τ')

    # capacity = largest λ where >= 75% of seeds are safe AND mean drop < 5%
    def capacity(rows):
        for r in reversed(rows):
            if r['safe_pct'] >= 75 and r['drop_mean'] < 5:
                return r['lam'], r['q_mean']
        return 0.0, 0.0
    lb, qb = capacity(base)
    lt, qt = capacity(tuned)
    print(f'\nRobust capacity (≥75% safe seeds, drops<5%):')
    print(f'  baseline τ:  λ ≤ {lb:.2f}  →  {qb*3600:.0f} veh/h')
    print(f'  tuned    τ:  λ ≤ {lt:.2f}  →  {qt*3600:.0f} veh/h')
    print(f'  signalised reference:           1800 veh/h')

    # ---- plot ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    lams = np.array(lambdas)

    ax = axes[0]
    for rows, color, marker, label in [
        (base, 'C0', 'o', 'baseline τ (c2_t=0.15, d_c=70)'),
        (tuned, 'C2', 's', 'tuned τ (c2_t=0.20, d_c=100)'),
    ]:
        q_mean = np.array([r['q_mean'] for r in rows]) * 3600
        q_p25 = np.array([r['q_p25'] for r in rows]) * 3600
        q_p75 = np.array([r['q_p75'] for r in rows]) * 3600
        ax.plot(lams, q_mean, color=color, marker=marker, label=f'{label} (mean)')
        ax.fill_between(lams, q_p25, q_p75, color=color, alpha=0.2)
    ax.plot([0, lams.max()], [0, 4 * lams.max() * 3600], ':', color='gray',
            label='4λ = no-drop demand')
    ax.axhline(1800, color='gray', linestyle='--', label='signalised ≈ 1800')
    ax.set_xlabel('λ per arm [veh/s]')
    ax.set_ylabel('realised q_out [veh/h]')
    ax.set_title(f'Throughput across seeds (N={N_SEEDS}, T={T_RUN}s)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for rows, color, marker, label in [
        (base, 'C0', 'o', 'baseline τ'),
        (tuned, 'C2', 's', 'tuned τ'),
    ]:
        med = np.array([r['intra_med'] for r in rows])
        p25 = np.array([r['intra_p25'] for r in rows])
        p75 = np.array([r['intra_p75'] for r in rows])
        ax.plot(lams, med, color=color, marker=marker, label=f'{label} (median)')
        ax.fill_between(lams, p25, p75, color=color, alpha=0.2)
    ax.axhline(M.D_A, color='gray', linestyle=':', label=f'd_a = {M.D_A:.0f}')
    ax.axhline(SAFETY, color='red', linestyle=':', label=f'd_a/2 = {SAFETY:.1f}')
    ax.set_xlabel('λ per arm [veh/s]')
    ax.set_ylabel('worst-case intra-min [m]')
    ax.set_title('Safety across seeds')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle(f'Exp D multi-seed diagnostic (N={N_SEEDS}, T={T_RUN}s)',
                 y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig('diagnose_intersection_mfd_seeds.png', dpi=110, bbox_inches='tight')
    print('\nsaved diagnose_intersection_mfd_seeds.png')


if __name__ == '__main__':
    main()
