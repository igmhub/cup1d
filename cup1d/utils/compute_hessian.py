"""
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
        for jj in range(nelem):
            if ii == jj:
                xhh = mod_elem(nelem, ii, hh)
                hessian[ii, jj] = (
                    func(p0 + xhh) + func(p0 - xhh) - 2 * func_p0
                ) / hh**2
            else:
                xhh = mod_elem(nelem, ii, hh)
                yhh = mod_elem(nelem, jj, hh)
                hessian[ii, jj] = (
                    func(p0 + xhh + yhh)
                    + func(p0 - xhh - yhh)
                    - func(p0 - xhh + yhh)
                    - func(p0 + xhh - yhh)
                ) / (4 * hh**2)

    return hessian
