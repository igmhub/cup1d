import numpy as np
from cup1d.models.contaminants.base_contaminants import Contaminant


def get_Rz(z, k_kms):
    # fig 32 https://arxiv.org/abs/2205.10939
    # lambda_AA = np.arange([3523.626, 3993.217, 4413.652, 4752.203, 5019.740, 5243.594, 5522.035, 5767.681, 5996.975, 6226.294, 6471.940, 6783.036])
    # resolution = np.array([2012.821, 2272.247, 2513.575, 2694.570, 2857.466, 2996.229, 3177.225, 3364.253, 3521.116, 3659.879, 3846.908, 4124.434])
    # rfit = np.polyfit(lambda_AA, resolution, 2)
    # plt.plot(lambda_AA, np.poly1d(rfit)(lambda_AA))

    """Estimate a velocity resolution width from the fitted resolving power.

    Parameters
    ----------
    z : float
        Redshift used to convert velocity to observed wavelength.
    k_kms : float or numpy.ndarray
        Wavenumber in s/km. The fit is evaluated at the corresponding
        wavelength scale ``2*pi/k_AA``.

    Returns
    -------
    float or numpy.ndarray
        Gaussian velocity width in km/s, using ``c/(2.355*resolving_power)``.
    """
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
    """Convert a fixed 0.8-Angstrom width into a velocity width.

    Parameters
    ----------
    z : float or numpy.ndarray
        Absorber redshift.

    Returns
    -------
    float or numpy.ndarray
        Width in km/s, equal to ``c*0.8/((1+z)*1215.67)``.
    """
    c_kms = 2.99792458e5
    lya_AA = 1215.67  # angstroms
    Delta_lambda_AA = 0.8  # angstroms
    # kms2AA = lya_AA * (1 + z) / c_kms
    # k_A = k_kms / kms2AA
    Rz = c_kms * Delta_lambda_AA / (1 + z) / lya_AA
    return Rz


class Resolution(Contaminant):
    """Model the multiplicative P1D correction for spectrograph resolution.

    The redshift-dependent ``R_coeff`` history controls the correction applied
    by ``get_contamination`` to velocity-space power spectra.
    """

    def __init__(
        self,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        z_max_res=10,
        fid_vals=None,
        flat_priors=None,
        null_vals=None,
        Gauss_priors=None,
    ):
        """Initialize the resolution-coefficient history and its priors.

        Parameters
        ----------
        coeffs, prop_coeffs, free_param_names
            Optional fixed history, metadata, and free coefficient selection.
        z_0 : float, default=3
            Pivot redshift of the coefficient polynomial.
        z_max_res : float, default=10
            Retained compatibility resolution cutoff.
        fid_vals, flat_priors, null_vals, Gauss_priors : mapping, optional
            History defaults and prior metadata forwarded to :class:`Contaminant`.
        """

        list_coeffs = ["R_coeff"]

        # priors for all coefficients
        if flat_priors is None:
            flat_priors = {"R_coeff": [[-0.5, 0.5], [-0.1, 0.1]]}

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

    def get_contamination(self, z, k_kms, like_params=None):
        """Evaluate multiplicative spectrograph-resolution corrections.

        Parameters
        ----------
        z : array-like
            Redshift rows.
        k_kms : sequence of ndarray
            Velocity wavenumber grids in s/km.
        like_params : mapping, optional
            Scalar resolution-coefficient overrides.

        Returns
        -------
        ndarray or list of ndarray
            Dimensionless correction factors.
        """

        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z, like_params=like_params)
            )
        # print(vals)

        cont = []
        for iz in range(len(z)):
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

    def get_contamination_batch(self, z, k_kms, like_params):
        """Evaluate resolution corrections for columnar coefficient samples.

        Returns one dimensionless ``(n_batch, nk_z)`` array per redshift.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        values = self.get_value_batch("R_coeff", z, like_params)
        return [
            1 + 2 * values[:, iz, None] * get_Rz_Naim(z[iz]) ** 2
            * np.asarray(k_kms[iz])[None, :] ** 2
            for iz in range(len(z))
        ]
