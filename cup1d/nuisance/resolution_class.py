import numpy as np
from cup1d.nuisance.base_contaminants import Contaminant


def get_Rz(z, k_kms):
    # fig 32 https://arxiv.org/abs/2205.10939
    # lambda_AA = np.arange([3523.626, 3993.217, 4413.652, 4752.203, 5019.740, 5243.594, 5522.035, 5767.681, 5996.975, 6226.294, 6471.940, 6783.036])
    # resolution = np.array([2012.821, 2272.247, 2513.575, 2694.570, 2857.466, 2996.229, 3177.225, 3364.253, 3521.116, 3659.879, 3846.908, 4124.434])
    # rfit = np.polyfit(lambda_AA, resolution, 2)
    # plt.plot(lambda_AA, np.poly1d(rfit)(lambda_AA))

    c_kms = 2.99792458e5
    lya_AA = 1215.67
    rfit = np.array([4.53087663e-05, 1.70716005e-01, 8.60679006e02])
    R_coeff_lambda = np.poly1d(rfit)
    kms2AA = lya_AA * (1 + z) / c_kms
    # lambda_kms = lambda_AA * AA2kms
    k_AA = k_kms / kms2AA
    lambda_AA = 2 * np.pi / k_AA

    Rz = c_kms / (2.355 * R_coeff_lambda(lambda_AA))

    return Rz


def get_Rz_Naim(z):
    # 4.1 https://arxiv.org/abs/2306.06316
    c_kms = 2.99792458e5
    lya_AA = 1215.67  # angstroms
    Delta_lambda_AA = 0.8  # angstroms
    # kms2AA = lya_AA * (1 + z) / c_kms
    # k_A = k_kms / kms2AA
    Rz = c_kms * Delta_lambda_AA / (1 + z) / lya_AA
    return Rz


class Resolution(Contaminant):
    """Use a handful of parameters to model the mean transmitted flux fraction
    (or mean flux) as a function of redshift.
     For now, we use a polynomial to describe log(tau_eff) around z_tau.
    """

    def __init__(
        self,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        z_max_res=3.8,
        fid_vals=None,
        flat_priors=None,
        null_vals=None,
        Gauss_priors=None,
    ):
        """Construct model as a rescaling around a fiducial mean flux"""

        list_coeffs = ["R_coeff"]

        # maximum redshift to apply correction
        self.z_max_res = z_max_res

        # priors for all coefficients
        if flat_priors is None:
            flat_priors = {"R_coeff": [[-0.5, 0.5], [-0.05, 0.05]]}

        # z dependence and output type of coefficients
        if prop_coeffs is None:
            prop_coeffs = {
                "R_coeff_ztype": "pivot",
                "R_coeff_otype": "const",
            }

        # fiducial values
        if (fid_vals is None) | (len(fid_vals["R_coeff"]) == 0):
            fid_vals = {
                "R_coeff": [0, 0],
            }

        super().__init__(
            coeffs=coeffs,
            list_coeffs=list_coeffs,
            prop_coeffs=prop_coeffs,
            free_param_names=free_param_names,
            z_0=z_0,
            fid_vals=fid_vals,
            null_vals=null_vals,
            flat_priors=flat_priors,
            Gauss_priors=Gauss_priors,
        )

    def get_contamination(self, z, k_kms, like_params=[]):
        """Multiplicative contamination caused by Resolution"""

        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z, like_params=like_params)
            )
        # print(vals)

        cont = []
        for iz in range(len(z)):
            if z[iz] > self.z_max_res:
                res = np.ones_like(k_kms[iz])
            else:
                res = (
                    1
                    + 2
                    * vals["R_coeff"][iz]
                    * get_Rz_Naim(z[iz]) ** 2
                    * k_kms[iz] ** 2
                )
            cont.append(res)

        if len(z) == 1:
            cont = cont[0]

        # print(cont)

        return cont
