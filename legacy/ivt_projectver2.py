"""Port of IVT Project/IVT_projectver2.m — early draft of the two-flock setup.

This is the broken-stub version: it has nested function definitions and an
empty inner loop preserved verbatim (the MATLAB `2:4:2` range yields a single
element 2, which translates to a single Python iteration at j=1 after the
1-to-0 index shift). The Dynamics/SigmaNorm/Neighbour helpers are shadowed
copies of the corresponding standalone modules, kept here for fidelity with
the original MATLAB file.

For the finished simulation see flocking_main.py.
"""
import numpy as np


def dynamics(a, teta, v, x, y, t, T):
    V = v[t] + T * a[t]
    X = x[t] + T * (v[t] * np.cos(teta[t]))
    Y = y[t] + T * (v[t] * np.sin(teta[t]))
    return V, X, Y


def sigma_norm(z, e):
    z = np.asarray(z, dtype=float)
    return (1.0 / e) * (np.sqrt(1.0 + e * np.dot(z, z)) - 1.0)


def neighbour(qref, qquery, r):
    qref = np.asarray(qref, dtype=float)
    qquery = np.asarray(qquery, dtype=float)
    return 1 if np.linalg.norm(qref - qquery) <= r else 0


def main():
    a_max = 9                              # max acceleration [m/s^2]
    v_max = 120 * 5 / 18                   # max speed [m/s]
    T = 50                                 # sim time [s]
    TS = 0.1                               # step time [s]
    N = 8                                  # number of cars
    N_lane = 4
    W_lane = 1.8
    l_y = N_lane * W_lane
    y_lim_up = l_y
    y_lim_dwn = 0
    r = 1
    d_flock = 100

    # Flock right initial conditions
    num_grids_flockR = 2
    x11_coor = 0.5
    y11_coor = 0.6

    # Flock left initial conditions
    num_grids_flockL = 2
    x21_coor = x11_coor - d_flock
    y21_coor = 0.6

    # populate flock — MATLAB inner range 2:4:2 yields [2]; that's j=1 in
    # 0-indexed Python.
    pR = np.zeros((num_grids_flockR, 2 * num_grids_flockR))
    for i in range(num_grids_flockR):
        for j in range(1, 2, 2 * num_grids_flockR):
            pR[i, :] = x11_coor + i * r
            pR[:, j] = y11_coor + j * r

    # NOTE: original MATLAB allocates pL but writes into pR. Preserved verbatim.
    pL = np.zeros((num_grids_flockR, 2 * num_grids_flockR))
    for i in range(num_grids_flockL):
        for j in range(1, 2, 2 * num_grids_flockL):
            pR[i, :] = x21_coor + i * r
            pR[:, j] = y21_coor + j * r

    # Agent movement variables
    v = np.zeros((N, int(T / TS)))
    x = np.zeros((N, int(T / TS)))
    y = np.zeros((N, int(T / TS)))

    # Control variables
    a = np.zeros((N, int(T / TS)))
    teta = np.zeros((N, int(T / TS)))

    # Smoke test from the original script:
    print(sigma_norm([1, 1, 1], 2))

    # Define agents
    #   Alpha agent
    #   Beta agent
    #   Gamma agent
    #   Tau agent

    # Road boundary setup
    # Plotting


if __name__ == '__main__':
    main()
