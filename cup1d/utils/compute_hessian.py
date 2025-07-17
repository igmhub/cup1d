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
