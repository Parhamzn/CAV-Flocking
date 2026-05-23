"""Unicycle-like movement dynamics (port of Dynamics.m).

V = v[t] + T * a[t];  X = x[t] + T * v[t] * cos(theta[t]);  Y = y[t] + T * v[t] * sin(theta[t]).

Not used by the alpha/beta/gamma/tau flocking driver, which uses double-integrator
dynamics. Kept for parity with the original MATLAB primitives.
"""
import numpy as np


def dynamics(a, theta, v, x, y, t, T):
    V = v[t] + T * a[t]
    X = x[t] + T * (v[t] * np.cos(theta[t]))
    Y = y[t] + T * (v[t] * np.sin(theta[t]))
    return V, X, Y
