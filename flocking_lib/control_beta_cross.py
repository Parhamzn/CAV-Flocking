"""β-agent for a 4-way cross intersection.

Road geometry is a "+" shape: horizontal road of half-width half_road
along y=0, vertical road of half-width half_road along x=0, meeting at a
central intersection box of half-width half_inter (typically = half_road).

Wall layout:
  * horizontal arms (|x| > half_inter, |y| ≤ half_road): walls at y=±half_road
  * vertical arms   (|y| > half_inter, |x| ≤ half_road): walls at x=±half_road
  * intersection box (|x| ≤ half_inter, |y| ≤ half_inter): no walls

Cars off-road (|x| > half_road AND |y| > half_road, the four outside
corners) get a hard inward push toward the origin.
"""
import numpy as np
from flocking_lib.sigma_norm import sigma_norm
from flocking_lib.rho import rho
from flocking_lib.phi_beta import phi_beta


def control_beta_cross(i, q, p, geom, params):
    qi = q[i]; pi = p[i]
    ui = np.zeros(2)
    half_road  = geom['half_road']
    half_inter = geom['half_inter']
    d_b = params['d_b']
    d_sigma = sigma_norm(d_b, params['e'])

    in_intersection = (abs(qi[0]) <= half_inter) and (abs(qi[1]) <= half_inter)
    if in_intersection:
        return ui                                        # no walls inside box

    in_horiz_arm = (abs(qi[1]) <= half_road) and (abs(qi[0]) > half_inter)
    in_vert_arm  = (abs(qi[0]) <= half_road) and (abs(qi[1]) > half_inter)

    if in_horiz_arm:
        for wall_y, inward in [(-half_road, +1.0), (+half_road, -1.0)]:
            abs_dist = abs(qi[1] - wall_y)
            z = sigma_norm(np.array([0.0, abs_dist]), params['e'])
            if z > d_sigma:
                continue
            mag = -phi_beta(z, d_b, params['h_b'], params['e'])
            ui[1] += params['c1_b'] * mag * inward
            bik = rho(z / d_sigma, params['h_b'])
            ui[1] += params['c2_b'] * bik * (0.0 - pi[1])

    if in_vert_arm:
        for wall_x, inward in [(-half_road, +1.0), (+half_road, -1.0)]:
            abs_dist = abs(qi[0] - wall_x)
            z = sigma_norm(np.array([abs_dist, 0.0]), params['e'])
            if z > d_sigma:
                continue
            mag = -phi_beta(z, d_b, params['h_b'], params['e'])
            ui[0] += params['c1_b'] * mag * inward
            bik = rho(z / d_sigma, params['h_b'])
            ui[0] += params['c2_b'] * bik * (0.0 - pi[0])

    if not in_horiz_arm and not in_vert_arm:
        # Off-road corner: hard pull back toward the nearest road arm.
        # Pick the nearest road by which axis the agent is closer to.
        if abs(qi[0]) < abs(qi[1]):
            # closer to vertical road — push horizontally toward x=0
            ui[0] += -params['c1_b'] * np.sign(qi[0])
        else:
            ui[1] += -params['c1_b'] * np.sign(qi[1])

    return ui
