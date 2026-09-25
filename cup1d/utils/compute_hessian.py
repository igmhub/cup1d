r"""
For convenience, we decided to use the inverse of the Hessian in order to get a first estimation of the error without doing the $\chi^2$ scan. We estimate it as follows.

\begin{itemize}
    \item I compute the the Hessian using finite differences. I am using the following expression for the diagonal elements
    \begin{equation}
        H[i, i] = [f(p + h) + f(p - h) - 2 * f(p)] / h^2
    \end{equation}
    and for the off-diagonal
    \begin{equation}
    H[i, j] = [f(p + h_x + h_y) + f(p - h_x - h_y) - f(p - h_x + h_y) - f(p + h_x - h_y)] / (4 * h^2)
    \end{equation}

    \item I then take the inverse of the matrix.

    \item The last step is that, since we are sampling $A_s$ and $n_s$ internally, I need to propagate errors into $\Delta^2_\star$ and $n_\star$.
"""


import numpy as np


def get_hessian(func, p0, hh=1e-4):
    def mod_elem(nelem, ind, val):
        xx = np.zeros(nelem)
        xx[ind] = val
        return xx

    nelem = len(p0)
    hessian = np.zeros((nelem, nelem))
    func_p0 = func(p0)
    for ii in range(nelem):
        xhh = mod_elem(nelem, ii, hh)
        hessian[ii, ii] = (
            func(p0 + xhh) + func(p0 - xhh) - 2 * func_p0
        ) / hh**2
        for jj in range(ii + 1, nelem):
            yhh = mod_elem(nelem, jj, hh)
            value = (
                func(p0 + xhh + yhh)
                + func(p0 - xhh - yhh)
                - func(p0 - xhh + yhh)
                - func(p0 + xhh - yhh)
            ) / (4 * hh**2)
            hessian[ii, jj] = value
            hessian[jj, ii] = value

    return hessian


def get_hessian_rows(func, p0, indices, hh=1e-4):
    """Return finite-difference Hessian rows for selected coordinates."""

    p0 = np.asarray(p0)
    hessian = np.zeros((len(p0), len(p0)))
    center = func(p0)
    for ii in indices:
        direction_i = np.zeros(len(p0))
        direction_i[ii] = hh
        hessian[ii, ii] = (
            func(p0 + direction_i) + func(p0 - direction_i) - 2 * center
        ) / hh**2
        for jj in range(len(p0)):
            if jj == ii:
                continue
            direction_j = np.zeros(len(p0))
            direction_j[jj] = hh
            hessian[ii, jj] = (
                func(p0 + direction_i + direction_j)
                + func(p0 - direction_i - direction_j)
                - func(p0 - direction_i + direction_j)
                - func(p0 + direction_i - direction_j)
            ) / (4 * hh**2)
    return hessian
