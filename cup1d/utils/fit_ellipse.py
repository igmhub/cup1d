import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def rho_from_axes(a, b, theta):
    """
    Compute correlation coefficient rho from ellipse semi-axes and angle.

    Parameters
    ----------
    a : float
        Semi-major axis length (any contour level).
    b : float
        Semi-minor axis length.
    theta : float
        Ellipse tilt angle in radians (major axis w.r.t. x-axis).

    Returns
    -------
    rho : float
        Correlation coefficient in [-1, 1].
    """
    num = (a**2 - b**2) * np.cos(theta) * np.sin(theta)
    den = np.sqrt(
        (a**2 * np.cos(theta) ** 2 + b**2 * np.sin(theta) ** 2)
        * (a**2 * np.sin(theta) ** 2 + b**2 * np.cos(theta) ** 2)
    )
    return num / den


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

    # correlation coefficient
    rho = rho_from_axes(a_len, b_len, theta)
    print("angle, rho:", theta * 180 / np.pi, rho)

    return xfit, yfit, rho


def plot_ellipse(sigma1=0.2, sigma2=0.5, rho=0.6, mean=[1.0, 2.0], ax=None):
    # Covariance matrix
    cov = np.array(
        [
            [sigma1**2, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2],
        ]
    )

    # Eigen-decomposition for ellipse axes
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # 68% chi-square value for 2 dof
    chi2_val = 2.30

    # Width and height of ellipse (2*sqrt because diameter)
    width, height = 2 * np.sqrt(eigvals * chi2_val)

    # Angle of ellipse (in degrees)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    # print("angle")
    # angle = np.degrees(0.14250882064032286) * 2

    # Plot
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor="r",
        facecolor="none",
        lw=2,
    )
    ax.add_patch(ellipse)

    ax.scatter(*mean, c="r", marker="x", label="Best fit")
