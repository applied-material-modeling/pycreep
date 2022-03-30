import numpy as np
import numpy.linalg as la

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
    """
    # Actually do the regression
    b, res, rank, svs = la.lstsq(X, y, rcond = None)

    # Predictions
    p = X.dot(b)

    # Error
    e = y - p

    # SSE
    n = len(y)
    N = np.eye(n) - np.ones((n,n))/n
    SSE = np.dot(p, np.dot(N, p))

    # R2
    SST = np.dot(y, np.dot(N, y))
    R2 = SSE/SST

    # SEE
    SEE = np.sqrt(np.sum(e**2.0) / (X.shape[0] - X.shape[1]))

    return b, p, SSE, R2, SEE

