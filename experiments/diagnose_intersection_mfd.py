"""Diagnostic for Exp D: sweep τ strength at a fixed demand, plus a
snapshot of the centre region so we can see whether the algorithm fails
by tight-radius collision at the centre, by arm queue back-up, or both.

Pick λ = 0.6/arm (3rd row of the main sweep) where baseline c2_t=0.15
collapses to intra=0.04. Sweep c2_t ∈ {0.10, 0.15, 0.20, 0.30, 0.45} and
d_c ∈ {50, 70, 100} and look at the (intra_min, throughput, drop%) grid.

A second pass takes the best (c2_t, d_c) combo found and re-runs the
full λ sweep with it.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import exp_intersection_mfd as M

ORIGINAL_C2T = M.C2_T
ORIGINAL_DC = M.D_C


def run_with_tau(lam, c2_t, d_c, T=40.0):
    M.C2_T = c2_t
    M.D_C = d_c
    return M.run_intersection_mfd(lam, T=T)


def main():
    print('--- τ tuning grid at λ = 0.6 veh/s/arm ---\n')
    print(f'{"c2_t":>6}  {"d_c":>5}  {"q_out":>7}  {"drop%":>6}  '
          f'{"intra":>6}  {"travel":>8}')
    lam_diag = 0.6
    grid = []
    for c2_t in [0.10, 0.15, 0.20, 0.30, 0.45]:
        for d_c in [50, 70, 100]:
            r = run_with_tau(lam_diag, c2_t, d_c)
            drop_pct = 100.0 * r['n_dropped'] / max(1, r['n_dropped'] + r['n_injected'])
            grid.append((c2_t, d_c, r))
            print(f'{c2_t:6.2f}  {d_c:5.0f}  {r["throughput_total"]:7.2f}  '
                  f'{drop_pct:5.1f}%  {r["min_pair_dist"]:6.2f}  '
                  f'{r["mean_travel_time"]:8.2f}')

    # pick the τ combo with the safest intra at lam_diag
    best = max(grid, key=lambda g: g[2]['min_pair_dist'])
    best_c2t, best_dc, best_r = best
    print(f'\nbest at λ=0.6: c2_t={best_c2t}, d_c={best_dc} '
          f'→ intra={best_r["min_pair_dist"]:.2f} m  '
          f'(baseline intra=0.04 m)\n')

    # re-run full λ sweep with the best τ combo and compare
    M.C2_T = best_c2t
    M.D_C = best_dc
    lambdas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    tuned_rows = [M.run_intersection_mfd(lam, T=60.0) for lam in lambdas]

    M.C2_T = ORIGINAL_C2T
    M.D_C = ORIGINAL_DC
    baseline_rows = [M.run_intersection_mfd(lam, T=60.0) for lam in lambdas]

    print(f'{"λ":>4}  {"q_out base":>10}  {"intra base":>10}  '
          f'{"q_out tuned":>11}  {"intra tuned":>11}')
    for lam, br, tr in zip(lambdas, baseline_rows, tuned_rows):
        print(f'{lam:4.2f}  {br["throughput_total"]:10.2f}  '
              f'{br["min_pair_dist"]:10.2f}  {tr["throughput_total"]:11.2f}  '
              f'{tr["min_pair_dist"]:11.2f}')

    # capacity in each
    def find_capacity(rows):
        for lam, r in reversed(list(zip(lambdas, rows))):
            drop_pct = 100.0 * r['n_dropped'] / max(1, r['n_dropped'] + r['n_injected'])
            if drop_pct < 5.0 and r['min_pair_dist'] > M.D_A / 2:
                return lam, r['throughput_total']
        return 0.0, 0.0
    lam_base, q_base = find_capacity(baseline_rows)
    lam_tuned, q_tuned = find_capacity(tuned_rows)
    print(f'\nbaseline capacity:  λ ≤ {lam_base:.2f}  ⇒ {q_base*3600:.0f} veh/h')
    print(f'tuned    capacity:  λ ≤ {lam_tuned:.2f}  ⇒ {q_tuned*3600:.0f} veh/h')

    # ---- plot --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lams = np.array(lambdas)
    ax = axes[0]
    ax.plot(lams, [r['throughput_total'] * 3600 for r in baseline_rows], '-o',
            color='C0', label=f'baseline τ (c2_t={ORIGINAL_C2T}, d_c={ORIGINAL_DC:.0f})')
    ax.plot(lams, [r['throughput_total'] * 3600 for r in tuned_rows], '-s',
            color='C2', label=f'tuned   τ (c2_t={best_c2t}, d_c={best_dc})')
    ax.plot([0, lams.max()], [0, 4 * lams.max() * 3600], ':', color='gray',
            label='4λ (no-drop demand)')
    ax.axhline(1800, color='gray', linestyle='--', label='signalised ≈ 1800')
    ax.set_xlabel('λ per arm [veh/s]')
    ax.set_ylabel('realised q_out [veh/h]')
    ax.set_title('Throughput')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(lams, [r['min_pair_dist'] for r in baseline_rows], '-o',
            color='C0', label=f'baseline τ')
    ax.plot(lams, [r['min_pair_dist'] for r in tuned_rows], '-s',
            color='C2', label=f'tuned τ')
    ax.axhline(M.D_A, color='gray', linestyle=':', label=f'd_a = {M.D_A:.0f}')
    ax.axhline(M.D_A / 2, color='red', linestyle=':',
               label=f'd_a/2 = {M.D_A/2:.1f} (safety)')
    ax.set_xlabel('λ per arm [veh/s]')
    ax.set_ylabel('worst-case intra-min [m]')
    ax.set_title('Safety')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle('Exp D diagnostic: τ tuning under continuous injection',
                 y=1.00)
    fig.tight_layout()
    fig.savefig('diagnose_intersection_mfd.png', dpi=110, bbox_inches='tight')
    print('\nsaved diagnose_intersection_mfd.png')


if __name__ == '__main__':
    main()
