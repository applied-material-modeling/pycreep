"""
Mathematical helper functions used in multiple modules
"""

import bisect

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


def asme_tensile_analysis(T, R, order, Tref=21.0):
    """
    Constrained polynomial regression to give the polynomial coefficients
    of an ASME-type tensile analysis

    Args:
        T:      temperatures
        R:      normalized stress
        order:  polynomial order

    Keyword Args:
        Tref:   reference temperature (default 21 C)

    Returns:
        p (np.array): polynomial coefficients
        R2 (float): coefficient of determination
    """
    x = T - Tref
    y = R - 1
    V = np.vander(x, order + 1)[:, :-1]

    p, _, _, R2, _ = least_squares(V, y)
    p = np.concatenate((p, [1.0]))  # Add the constant term

    return p, R2


def find_nearest_index(sorted_list, target):
    """
    Finds the index of the element closest to the target in a sorted list.

    Args:
        sorted_list (list): A list of numbers sorted in ascending order.
        target (int or float): The value to find the nearest element to.

    Returns:
        int: The index of the nearest element in the sorted list.
    """
    # Find the insertion point for the target
    # bisect_left returns an index where the target could be inserted
    # to maintain sorted order, and all elements to its left are < target.
    # bisect_right returns an index where the target could be inserted,
    # and all elements to its left are <= target.
    # For finding the *nearest* value, bisect_left is often preferred as a starting point.
    idx = bisect.bisect_left(sorted_list, target)

    if idx == 0:  # Target is smaller than or equal to the first element
        return 0
    if idx == len(sorted_list):  # Target is larger than the last element
        return len(sorted_list) - 1

    # Compare the element at 'idx' and the element before it ('idx-1')
    # to find which is closer to the target.
    before = sorted_list[idx - 1]
    after = sorted_list[idx]

    if abs(target - before) <= abs(target - after):
        return idx - 1
    return idx
