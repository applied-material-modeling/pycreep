"""
Mathematical helper functions used in multiple modules
"""

import numpy as np
import numpy.linalg as la
import scipy.optimize as opt


def least_squares(X, y):
    """
    Expanded least squares regression routine

    Args:
        X:      overdetermined system
        y:      RHS

    Returns:
        b:      best fit
        SSE:    standard squared error
        R2:     Coefficient of determination
        SEE:    Standard error estimate
    """
    # Actually do the regression
    b, _, _, _ = la.lstsq(X, y, rcond=None)

    # Predictions
    p = X.dot(b)

    # Error
    e = y - p

    # SSE
    n = len(y)
    N = np.eye(n) - np.ones((n, n)) / n
    SSE = np.dot(p, np.dot(N, p))

    # R2
    SST = np.dot(y, np.dot(N, y))
    R2 = SSE / SST

    # SEE
    SEE = np.sqrt(np.sum(e**2.0) / (X.shape[0] - X.shape[1]))

    return b, p, SSE, R2, SEE


def polynomial_fit(x, y, deg):
    """
    Polynomial regression with the more accurate routines

    Args:
        x:      inputs
        y:      outputs
        deg:    polynomial degree

    Returns:
        b:      best fit
        SSE:    standard squared error
        R2:     Coefficient of determination
        SEE:    Standard error estimate
    """
    b = np.polyfit(x, y, deg)

    # Predictions
    p = np.polyval(b, x)

    # Error
    e = y - p

    # SSE
    SSE = np.dot(p, p - np.mean(p))

    # R2
    SST = np.dot(y, y - np.mean(y))
    R2 = SSE / SST

    # SEE
    SEE = np.sqrt(np.sum(e**2.0) / (len(x) - (deg + 1)))

    return b, p, SSE, R2, SEE


def optimize_polynomial_fit(x, y, deg, X0, bounds, map_fn):
    """
    Change the values of X0 to find the optimal regression
    between map_fn(x, X0) and y
    """

    def fn(X):
        return -polynomial_fit(map_fn(x, X), y, deg)[3]

    res = opt.minimize(fn, X0, method="L-BFGS-B", bounds=bounds)

    return (res.x,) + polynomial_fit(map_fn(x, res.x), y, deg)
