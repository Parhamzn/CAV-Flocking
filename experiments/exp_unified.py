"""Combine the three Phase-2 improvements into a unified "V2" algorithm and
benchmark across all the scenarios from Phase 1.

V0 = baseline (Phase 1 final):
    velocity-only McKenzie τ (c1_t=0, c2_t=0.08)
    + corrected β (always-inward)
    + velocity-feedback γ
    + matrix J=[[0,1],[1,0]] (McKenzie reflection)

V2 = unified improved algorithm:
    velocity-only τ
    + corrected β
    + velocity γ  + OPTIONAL position γ toward externally-assigned target bands
    + predictive suppression (threshold = 10 m)
    + matrix R(+90°)=[[0,-1],[1,0]] (true rotation)

For corridor scenarios we test both with and without target γ. For
intersection scenarios we test V2 without target γ (no natural lane).
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from flocking_lib.flock_layouts import grid_formation
from flocking_lib.multi_flock_sim import run_multi_flock, encounter_metrics
from flocking_lib.control_tau import J, R_PLUS_90
from exp_intersection import make_flock_at_angle, metrics_intersection


def v0_params():
    return {
        'e':0.1,'a':5,'b':5,
        'd_a':7,'r_a':1.2*7,'h_a':0.2,
        'c1_a':5,'c2_a':2*np.sqrt(5),
        'd_b':3.0,'h_b':0.2,'c1_b':200,'c2_b':2*np.sqrt(200),
        'c_g':1.5,
        'd_c':40.0,'c1_t':0.0,'c2_t':0.08,
        # tau_matrix=J by default, no suppression, no target γ.
    }


def v2_params(y_target_per_flock=None, c_g_pos=0.5,
              suppress_th=10.0, c2_t=None, d_c=None):
    p = v0_params()
    p['tau_matrix'] = R_PLUS_90
    p['predict_suppress_threshold'] = suppress_th
    if y_target_per_flock is not None:
        p['y_target_per_flock'] = y_target_per_flock
        p['c_g_pos'] = c_g_pos
    if c2_t is not None:
        p['c2_t'] = c2_t
    if d_c is not None:
        p['d_c'] = d_c
    return p


def metrics_corridor(q, flock_id, y_hi, d_b):
    return encounter_metrics(q, flock_id, y_hi, d_b)


# ---- Scenario runners ---------------------------------------------------
def run_corridor_2flock(rows, cols, params, y_hi=14.4, T=12.0,
                        flock_centers=(7.2, 7.2), p_d_dict=None):
    s = params['d_a']
    v1 = np.array([+10.0, 0.0]); v2 = np.array([-10.0, 0.0])
    q1, p1 = grid_formation(rows, cols,
                            x_center=-20-(cols-1)*s/2, y_center=flock_centers[0],
                            spacing=s, vel=v1)
    q2, p2 = grid_formation(rows, cols,
                            x_center=+20+(cols-1)*s/2, y_center=flock_centers[1],
                            spacing=s, vel=v2)
    p = dict(params, p_d_per_flock={1: v1, 2: v2})
    return run_multi_flock([(q1, p1), (q2, p2)], p,
                            y_lo=0.0, y_hi=y_hi, T=T, TS=0.02)


def run_intersection(angles, params, dist=60.0, T=18.0, y_extent=200.0):
    flock_inits = []
    p_d_dict = {}
    for k, ang in enumerate(angles, start=1):
        q, p = make_flock_at_angle(4, ang, dist, params['d_a'], 10.0)
        flock_inits.append((q, p))
        p_d_dict[k] = p[0].copy()
    pp = dict(params, p_d_per_flock=p_d_dict)
    return run_multi_flock(flock_inits, pp,
                            y_lo=-y_extent, y_hi=+y_extent, T=T, TS=0.02)


def main():
    print(f'{"scenario":<42}  {"V0 inter":>9}  {"V2 inter":>9}  {"Δ":>7}')
    print('-' * 80)

    # --- 1. Canonical 2-flock head-on (regression check) ----------------
    p0 = v0_params(); p2 = v2_params()
    q,_,_,fid = run_corridor_2flock(1, 4, p0)
    inter_0 = metrics_corridor(q, fid, 14.4, p0['d_b'])['inter_min']
    q,_,_,fid = run_corridor_2flock(1, 4, p2)
    inter_2 = metrics_corridor(q, fid, 14.4, p2['d_b'])['inter_min']
    print(f'{"1×4 vs 1×4 head-on (canonical)":<42}  {inter_0:8.2f}m  {inter_2:8.2f}m  {inter_2-inter_0:+7.2f}')

    # --- 2. Compression scenario D --------------------------------------
    p0 = v0_params()
    q,_,_,fid = run_corridor_2flock(2, 4, p0, y_hi=24.0, flock_centers=(12.0, 12.0))
    m_0 = metrics_corridor(q, fid, 24.0, p0['d_b'])
    p2 = v2_params(y_target_per_flock={1: 5.0, 2: 19.0})
    q,_,_,fid = run_corridor_2flock(2, 4, p2, y_hi=24.0, flock_centers=(12.0, 12.0))
    m_2 = metrics_corridor(q, fid, 24.0, p2['d_b'])
    print(f'{"2×4 vs 2×4 (compression D)":<42}  {m_0["inter_min"]:8.2f}m  {m_2["inter_min"]:8.2f}m  {m_2["inter_min"]-m_0["inter_min"]:+7.2f}    intra: {m_0["intra_min"]:5.2f} → {m_2["intra_min"]:5.2f}')

    # --- 3. v_d=3 (slow-end failure) ------------------------------------
    p0 = v0_params(); p0['p_d_flock1'] = np.array([+3.0, 0.0]); p0['p_d_flock2'] = np.array([-3.0, 0.0])
    p2 = v2_params(); p2['p_d_flock1'] = np.array([+3.0, 0.0]); p2['p_d_flock2'] = np.array([-3.0, 0.0])
    for label, params in [('v_d=3', p0), ('v_d=3 (V2)', p2)]:
        pass
    s = p0['d_a']
    q1, p1 = grid_formation(1, 4, x_center=-20-1.5*s, y_center=7.2, spacing=s, vel=p0['p_d_flock1'])
    q2, p2_init = grid_formation(1, 4, x_center=+20+1.5*s, y_center=7.2, spacing=s, vel=p0['p_d_flock2'])
    p0['p_d_per_flock'] = {1: p0['p_d_flock1'], 2: p0['p_d_flock2']}
    p2['p_d_per_flock'] = {1: p2['p_d_flock1'], 2: p2['p_d_flock2']}
    q,_,_,fid = run_multi_flock([(q1, p1), (q2, p2_init)], p0, y_lo=0.0, y_hi=14.4, T=22.0, TS=0.02)
    inter_v0_slow = metrics_corridor(q, fid, 14.4, p0['d_b'])['inter_min']
    q,_,_,fid = run_multi_flock([(q1, p1), (q2, p2_init)], p2, y_lo=0.0, y_hi=14.4, T=22.0, TS=0.02)
    inter_v2_slow = metrics_corridor(q, fid, 14.4, p2['d_b'])['inter_min']
    print(f'{"v_d=3 (slow end failure)":<42}  {inter_v0_slow:8.2f}m  {inter_v2_slow:8.2f}m  {inter_v2_slow-inter_v0_slow:+7.2f}')

    # --- 4. v_d=25 (fast-end failure) -----------------------------------
    p0 = v0_params(); p0['p_d_flock1'] = np.array([+25.0, 0.0]); p0['p_d_flock2'] = np.array([-25.0, 0.0])
    p2 = v2_params(); p2['p_d_flock1'] = np.array([+25.0, 0.0]); p2['p_d_flock2'] = np.array([-25.0, 0.0])
    q1, p1 = grid_formation(1, 4, x_center=-20-1.5*s, y_center=7.2, spacing=s, vel=p0['p_d_flock1'])
    q2, p2_init = grid_formation(1, 4, x_center=+20+1.5*s, y_center=7.2, spacing=s, vel=p0['p_d_flock2'])
    p0['p_d_per_flock'] = {1: p0['p_d_flock1'], 2: p0['p_d_flock2']}
    p2['p_d_per_flock'] = {1: p2['p_d_flock1'], 2: p2['p_d_flock2']}
    q,_,_,fid = run_multi_flock([(q1, p1), (q2, p2_init)], p0, y_lo=0.0, y_hi=14.4, T=8.0, TS=0.02)
    inter_v0_fast = metrics_corridor(q, fid, 14.4, p0['d_b'])['inter_min']
    esc_v0 = int(((q[:,1,:] < 0) | (q[:,1,:] > 14.4)).sum())
    q,_,_,fid = run_multi_flock([(q1, p1), (q2, p2_init)], p2, y_lo=0.0, y_hi=14.4, T=8.0, TS=0.02)
    inter_v2_fast = metrics_corridor(q, fid, 14.4, p2['d_b'])['inter_min']
    esc_v2 = int(((q[:,1,:] < 0) | (q[:,1,:] > 14.4)).sum())
    print(f'{"v_d=25 (fast end failure)":<42}  {inter_v0_fast:8.2f}m  {inter_v2_fast:8.2f}m  {inter_v2_fast-inter_v0_fast:+7.2f}    escapes: {esc_v0} → {esc_v2}')

    # --- 5. Offset sweep dy=14 (wasted τ) -------------------------------
    p0 = v0_params(); p2 = v2_params()
    v1 = np.array([+10.0, 0.0]); v2v = np.array([-10.0, 0.0])
    s = p0['d_a']
    q1, p1 = grid_formation(1, 4, x_center=-20-1.5*s, y_center=5.0, spacing=s, vel=v1)
    q2, p2_init = grid_formation(1, 4, x_center=+20+1.5*s, y_center=19.0, spacing=s, vel=v2v)
    p0['p_d_per_flock'] = {1: v1, 2: v2v}; p2['p_d_per_flock'] = {1: v1, 2: v2v}
    q,_,_,fid = run_multi_flock([(q1, p1), (q2, p2_init)], p0, y_lo=0.0, y_hi=24.0, T=12.0, TS=0.02)
    inter_v0_off = metrics_corridor(q, fid, 24.0, p0['d_b'])['inter_min']
    q,_,_,fid = run_multi_flock([(q1, p1), (q2, p2_init)], p2, y_lo=0.0, y_hi=24.0, T=12.0, TS=0.02)
    inter_v2_off = metrics_corridor(q, fid, 24.0, p2['d_b'])['inter_min']
    print(f'{"dy=14 offset (wasted τ)":<42}  {inter_v0_off:8.2f}m  {inter_v2_off:8.2f}m  {inter_v2_off-inter_v0_off:+7.2f}')

    # --- 6. 3-flock at 120° ---------------------------------------------
    p0 = v0_params(); p2 = v2_params(c2_t=0.15, d_c=70.0)
    q,_,_,fid = run_intersection([0, 120, 240], p0)
    m_3v0 = metrics_intersection(q, _, fid, 0.02, 10.0)
    q,_,_,fid = run_intersection([0, 120, 240], p2)
    m_3v2 = metrics_intersection(q, _, fid, 0.02, 10.0)
    print(f'{"3-flock at 120° (intersection)":<42}  {m_3v0["inter_min"]:8.2f}m  {m_3v2["inter_min"]:8.2f}m  {m_3v2["inter_min"]-m_3v0["inter_min"]:+7.2f}')

    # --- 7. 4-flock at 90° symmetric ------------------------------------
    p0 = v0_params(); p2 = v2_params(c2_t=0.15, d_c=70.0)
    q,_,_,fid = run_intersection([0, 90, 180, 270], p0)
    m_4v0 = metrics_intersection(q, _, fid, 0.02, 10.0)
    q,_,_,fid = run_intersection([0, 90, 180, 270], p2)
    m_4v2 = metrics_intersection(q, _, fid, 0.02, 10.0)
    print(f'{"4-flock at 90° symmetric":<42}  {m_4v0["inter_min"]:8.2f}m  {m_4v2["inter_min"]:8.2f}m  {m_4v2["inter_min"]-m_4v0["inter_min"]:+7.2f}')


if __name__ == '__main__':
    main()
