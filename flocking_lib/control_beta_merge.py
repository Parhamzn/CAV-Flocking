"""β-agent for a merge geometry: main road + on-ramp + merge zone.

Layout (with default geom):
  * Main road: y in [0, 7], all x.
  * On-ramp pre-merge (x < 0): y in [-7, 0], separated from main by a gore
    divider at y=0.
  * Merge zone (0 ≤ x ≤ L_merge): the gore disappears, and the on-ramp's
    outer (south) wall ramps linearly from y=-7 (at x=0) up to y=0 (at
    x=L_merge). The on-ramp narrows and disappears.
  * Post-merge (x > L_merge): main road only (y in [0, 7]).

At any agent position (x, y) we determine the local top and bottom walls
and apply β-repulsion as in the corrected control_beta. Agents that wander
off-road (y < bot_wall or y > top_wall) still get the inward push because
the corrected β logic always pushes toward the road.
"""
import numpy as np
from flocking_lib.sigma_norm import sigma_norm
from flocking_lib.rho import rho
from flocking_lib.phi_beta import phi_beta


def _apply_horizontal_wall(qi, pi, wall_y, inward_y, params, d_sigma):
    """Inward-only β-force for a horizontal wall at y = wall_y."""
    abs_dist = abs(qi[1] - wall_y)
    z = sigma_norm(np.array([0.0, abs_dist]), params['e'])
    if z > d_sigma:
        return np.zeros(2)
    mag = -phi_beta(z, params['d_b'], params['h_b'], params['e'])
    out = np.zeros(2)
    out[1] += params['c1_b'] * mag * inward_y
    bik = rho(z / d_sigma, params['h_b'])
    out[1] += params['c2_b'] * bik * (0.0 - pi[1])
    return out


def control_beta_merge(i, q, p, geom, params):
    qi = q[i]; pi = p[i]
    x, y = qi[0], qi[1]
    L = geom['L_merge']
    main_top = geom.get('main_top', 7.0)
    main_bot = geom.get('main_bot', 0.0)
    ramp_bot = geom.get('ramp_bot', -7.0)

    # Determine local top and bottom walls.
    if x < 0:
        # Two-lane pre-merge: main road or on-ramp depending on y side of gore.
        if y >= 0:
            top_wall = main_top
            bot_wall = main_bot          # gore at y = 0
        else:
            top_wall = main_bot          # gore at y = 0 (acting as top of on-ramp)
            bot_wall = ramp_bot
    elif x <= L:
        # Merge zone: bottom wall ramps from ramp_bot (at x=0) to main_bot (at x=L).
        frac = x / L
        bot_wall = ramp_bot + (main_bot - ramp_bot) * frac
        top_wall = main_top
    else:
        # Post-merge: main road only.
        top_wall = main_top
        bot_wall = main_bot

    d_sigma = sigma_norm(params['d_b'], params['e'])
    ui = np.zeros(2)
    ui += _apply_horizontal_wall(qi, pi, top_wall, -1.0, params, d_sigma)
    ui += _apply_horizontal_wall(qi, pi, bot_wall, +1.0, params, d_sigma)
    return ui
