"""f_beta for agent i (port of ControlBeta.m).

Interaction with road boundaries (top and bottom walls). Each wall is a
horizontal line; the beta-agent's job is to push the alpha-agent back
toward the road centerline whenever it gets within d_b of the wall.

NOTE on the geometry fix vs the original MATLAB port:

  The first port used Olfati-Saber's beta-agent projection literally:
  q_beta = (x_i, wall_y), n_ij = (q_beta - q_i)/sqrt(...). That works
  fine when the agent is inside the road (it gets pushed away from the
  wall), but the *direction* of n_ij flips when the agent crosses the
  wall, so the force flips with it and ends up pushing the escaped agent
  *further away from the road*. That's a bug — for hard barriers we want
  the force to be unconditionally inward.

  The fix below removes the projection-via-n_ij and instead applies the
  beta-repulsion magnitude in a hard-coded inward direction per wall:
  +y for the lower wall, -y for the upper wall. The velocity-damping
  term (consensus with the stationary wall) only acts on the y-axis.
"""
import numpy as np
from flocking_lib.sigma_norm import sigma_norm
from flocking_lib.rho import rho
from flocking_lib.phi_beta import phi_beta


def control_beta(i, q, p, y_lo, y_hi, params):
    qi = q[i]
    pi = p[i]
    ui = np.zeros(2)
    d_sigma = sigma_norm(params['d_b'], params['e'])

    for wall_y, inward_y in [(y_lo, +1.0), (y_hi, -1.0)]:
        # Absolute y-distance from this wall.
        abs_dist = abs(qi[1] - wall_y)
        # Sigma-norm of the offset vector (0, abs_dist).
        z = sigma_norm(np.array([0.0, abs_dist]), params['e'])
        if z > d_sigma:
            continue
        # phi_beta is repulsive (returns negative). Use its magnitude and
        # apply in the fixed inward direction — works on both sides of the wall.
        force_mag = -phi_beta(z, params['d_b'], params['h_b'], params['e'])
        ui[1] += params['c1_b'] * force_mag * inward_y
        # Velocity-matching to a stationary wall: damp y-velocity toward zero.
        bik = rho(z / d_sigma, params['h_b'])
        ui[1] += params['c2_b'] * bik * (0.0 - pi[1])
    return ui
