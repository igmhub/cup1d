import numpy as np
import copy
import os
import lace
from scipy.interpolate import interp1d
from cup1d.likelihood import likelihood_parameter
from cup1d.nuisance.resolution_model import get_Rz


def split_into_n_chunks(array, num_chunks):
    # Compute base chunk size and remainder
    chunk_size, remainder = divmod(len(array), num_chunks)

    # Create chunks and indexes
    chunks = []
    chunk_indexes = []
    start_idx = 0

    for i in range(num_chunks):
        # Distribute the remainder to make chunks as equal as possible
        extra = 1 if i < remainder else 0
        end_idx = start_idx + chunk_size + extra
        chunks.append(array[start_idx:end_idx])
        chunk_indexes.append(list(range(start_idx, end_idx)))
        start_idx = end_idx

    # Compute central values
    central_values = [np.median(chunk) for chunk in chunks]

    return central_values, chunk_indexes


class Resolution_Model_Chunks(object):
    """Use a handful of parameters to model the mean transmitted flux fraction
    (or mean flux) as a function of redshift.
     For now, we use a polynomial to describe log(tau_eff) around z_tau.
    """

    def __init__(
        self,
        R_coeff=None,
        free_param_names=None,
        fid_R_coeff=None,
        prior_width=1.5,
    ):
        """Construct model as a rescaling around a fiducial mean flux"""

        # Gaussian prior, translate to flat prior * 3 sigma
        self.prior_width = prior_width * 3

        if R_coeff is not None:
            if free_param_names is not None:
                raise ValueError("can not specify coeff and free_param_names")
            self.R_coeff = R_coeff
        else:
            if free_param_names:
                # figure out number of Resolution_Model free params
                n_R = len([p for p in free_param_names if "R_coeff_" in p])
                if n_R == 0:
                    n_R = 1
            else:
                n_R = 1

            self.R_coeff = np.zeros(n_R)

        self.n_R = len(self.R_coeff)

        self.set_parameters()

    def set_parameters(self):
        """Setup likelihood parameters in the mean flux model"""

        self.params = []
        Npar = len(self.R_coeff)
        for ii in range(Npar):
            name = "R_coeff_" + str(ii)
            value = self.R_coeff[ii]
            par = likelihood_parameter.LikelihoodParameter(
                name=name,
                value=value,
                min_value=-self.prior_width,
                max_value=self.prior_width,
            )
            self.params.append(par)

        return

    def get_Nparam(self):
        """Number of parameters in the model"""
        assert len(self.R_coeff) == len(self.params), "size mismatch"
        return len(self.R_coeff)

    def get_contamination(self, z, k_kms, like_params=[]):
        """Multiplicative contamination caused by Resolution"""

        R_coeff = self.get_R_coeffs(like_params=like_params)

        z_pivot, z_pivot_index = split_into_n_chunks(z, self.n_R)

        cont = []
        for ii in range(self.n_R):
            res = (
                1
                + 1e-2
                * R_coeff[ii]
                * get_Rz(z_pivot[ii], k_kms[ii]) ** 2
                * k_kms[ii] ** 2
            )
            cont.append(res)

        return cont

    def get_parameters(self):
        """Return likelihood parameters for the mean flux model"""
        return self.params

    def get_R_coeffs(self, like_params=[]):
        """Return list of mean flux coefficients"""

        if like_params:
            R_coeff = self.R_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "R_coeff_" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            if Npar != len(self.params):
                raise ValueError(
                    f"number of params {Npar} vs {self.params} mismatch in get_tau_coeffs"
                )

            for ip in range(Npar):
                _ = np.argwhere(self.params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + self.params[ip].name
                    )
                else:
                    R_coeff[ip] = array_values[_[0]]
        else:
            R_coeff = self.R_coeff

        return R_coeff
