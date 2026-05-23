"""Initial-condition builders for arbitrary flock geometries.

Each layout returns (q_init, p_init), an (N, 2) position array and an (N, 2)
velocity array. Velocities are uniform within a flock (the desired travel
direction). All formations are built so that nearest-neighbour spacing matches
d_a — keeps the alpha-lattice at rest at t=0 regardless of shape.
"""
import numpy as np


def grid_formation(rows, cols, x_center, y_center, spacing, vel):
    """Rectangular grid of `rows` x `cols` cars centred at (x_center, y_center).

    Cars are spaced `spacing` apart in both x and y.
    """
    N = rows * cols
    q = np.zeros((N, 2))
    # Build relative to (0, 0), then shift.
    xs = (np.arange(cols) - (cols - 1) / 2.0) * spacing
    ys = (np.arange(rows) - (rows - 1) / 2.0) * spacing
    X, Y = np.meshgrid(xs, ys)
    q[:, 0] = X.ravel() + x_center
    q[:, 1] = Y.ravel() + y_center
    p = np.tile(vel, (N, 1))
    return q, p


def line_formation(N, x_center, y_center, spacing, vel, axis='x'):
    """Single line of N cars centred at (x_center, y_center) along `axis`."""
    if axis == 'x':
        return grid_formation(rows=1, cols=N,
                              x_center=x_center, y_center=y_center,
                              spacing=spacing, vel=vel)
    elif axis == 'y':
        return grid_formation(rows=N, cols=1,
                              x_center=x_center, y_center=y_center,
                              spacing=spacing, vel=vel)
    else:
        raise ValueError("axis must be 'x' or 'y'")
