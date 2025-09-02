import numpy as np


def fit_ellipse(x, y, npts=200):
    """
    Fit an ellipse to scattered (x, y) points, ignoring NaNs.
    Returns parametric fit (xfit, yfit).
    """
    # remove NaNs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    # design matrix for conic fit
    D = np.vstack([x**2, x * y, y**2, x, y, np.ones_like(x)]).T
    S = np.dot(D.T, D)

    # constraint matrix
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1

    # solve generalized eigenvalue problem
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S).dot(C))
    a = eigvecs[:, np.argmax(eigvals.real)]

    # ellipse parameters
    A, B, Cc, Dd, Ee, Ff = a
    # center
    num = B**2 - 4 * A * Cc
    x0 = (2 * Cc * Dd - B * Ee) / num
    y0 = (2 * A * Ee - B * Dd) / num

    # orientation
    theta = 0.5 * np.arctan2(B, A - Cc)

    # axes lengths
    up = 2 * (A * x0**2 + B * x0 * y0 + Cc * y0**2 + Dd * x0 + Ee * y0 + Ff)
    down1 = (A + Cc) + np.sqrt((A - Cc) ** 2 + B**2)
    down2 = (A + Cc) - np.sqrt((A - Cc) ** 2 + B**2)
    a_len = np.sqrt(abs(up / down1))
    b_len = np.sqrt(abs(up / down2))

    # parametric fit
    t = np.linspace(0, 2 * np.pi, npts)
    xfit = (
        x0
        + a_len * np.cos(t) * np.cos(theta)
        - b_len * np.sin(t) * np.sin(theta)
    )
    yfit = (
        y0
        + a_len * np.cos(t) * np.sin(theta)
        + b_len * np.sin(t) * np.cos(theta)
    )

    params = {
        "x0": x0,
        "y0": y0,
        "a": a_len,
        "b": b_len,
        "theta": theta,
    }

    return xfit, yfit, params
