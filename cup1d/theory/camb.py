import numpy as np
from lace.cosmo.cosmology import Cosmology
from cup1d.likelihood import parameter as likelihood_parameter


class CAMBModel(object):
    """Interface between CAMB object and Theory"""

    def __init__(
        self, zs, cosmo=None, z_star=3.0, kp_kms=0.009, fast_camb=True
    ):
        """Setup from CAMB object and list of redshifts"""

        # list of redshifts at which we evaluate linear power
        self.zs = zs
        self.fast_camb = fast_camb

        # setup CAMB cosmology object
        if cosmo is None:
            self.cosmo = Cosmology()
        else:
            self.cosmo = cosmo

        # cache CAMB results when computed
        self.cached_camb_results = None
        # cache wavenumbers and linear power (at zs) when computed
        self.cached_linP_Mpc = None
        # cache linear power parameters at (z_star, kp_kms)
        self.z_star = z_star
        self.kp_kms = kp_kms
        self.cached_linP_params = None

    def get_likelihood_parameters(self, cosmo_priors=None):
        """Return a list of likelihood parameters"""

        # should clarify role of min/max given that these are also
        # set in the likelihood

        background = self.cosmo.get_background_params()
        primordial = self.cosmo.get_primordial_params()
        params = []
        params.append(
            likelihood_parameter.make_parameter(
                name="ombh2",
                min_value=0.018,
                max_value=0.026,
                value=background["ombh2"],
            )
        )
        params.append(
            likelihood_parameter.make_parameter(
                name="omch2",
                min_value=0.10,
                max_value=0.14,
                value=background["omch2"],
            )
        )
        if cosmo_priors is not None:
            min_val = cosmo_priors["As"][0]
            max_val = cosmo_priors["As"][1]
        else:
            min_val = 0.90e-09
            max_val = 3.60e-09
        # print(min_val, max_val)
        params.append(
            likelihood_parameter.make_parameter(
                name="As",
                min_value=min_val,
                max_value=max_val,
                value=primordial["As"],
            )
        )

        if cosmo_priors is not None:
            min_val = cosmo_priors["ns"][0]
            max_val = cosmo_priors["ns"][1]
        else:
            min_val = 0.85
            max_val = 1.10
        # print(min_val, max_val)
        params.append(
            likelihood_parameter.make_parameter(
                name="ns",
                min_value=min_val,
                max_value=max_val,
                value=primordial["ns"],
            )
        )
        params.append(
            likelihood_parameter.make_parameter(
                name="mnu",
                min_value=0.0,
                max_value=1.0,
                value=self.cosmo.get_mnu(),
            )
        )

        if cosmo_priors is not None:
            min_val = cosmo_priors["nrun"][0]
            max_val = cosmo_priors["nrun"][1]
        else:
            min_val = -0.05
            max_val = 0.05
        params.append(
            likelihood_parameter.make_parameter(
                name="nrun",
                min_value=min_val,
                max_value=max_val,
                value=primordial["nrun"],
            )
        )
        params.append(
            likelihood_parameter.make_parameter(
                name="H0", min_value=50, max_value=100, value=self.cosmo.get_H0()
            )
        )

        return {parameter["name"]: parameter for parameter in params}

    def get_camb_results(self):
        """Check if we have called CAMB.get_results yet, to save time.
        It returns a CAMB.results object."""

        if self.cached_camb_results is None:
            self.cached_camb_results = self.cosmo.get_CAMBdata()

        return self.cached_camb_results

    def get_linP_Mpc(self):
        """Check if we have already computed linP_Mpc, to save time.
        It returns (k_Mpc, zs, linP_Mpc)."""

        if self.cached_linP_Mpc is None:
            k_Mpc = np.logspace(-4, np.log10(self.cosmo.get_kmax_linP_Mpc()), 1000)
            linP_Mpc = self.cosmo.get_linP_Mpc(np.asarray(self.zs), k_Mpc)
            self.cached_linP_Mpc = (k_Mpc, list(self.zs), linP_Mpc)

        return self.cached_linP_Mpc

    def get_linP_params(self):
        """Linear power parameters at (z_star,kp_kms) for this cosmology"""

        if self.cached_linP_params is None:
            self.cached_linP_params = self.cosmo.get_linP_kms_params(
                self.z_star, self.kp_kms
            )

        return self.cached_linP_params

    def get_linP_Mpc_params(self, kp_Mpc):
        """Get linear power parameters to call emulator, at each z.
        Amplitude, slope and running around pivot point kp_Mpc."""

        return [
            self.cosmo.get_linP_Mpc_params(z, kp_Mpc) for z in self.zs
        ]

    def dkms_dMpc(self, z):
        """Return H(z)/(1+z) to convert Mpc to km/s"""

        return self.cosmo.get_dkms_dMpc(z)

    def get_M_of_zs(self):
        """Return M(z)=H(z)/(1+z) for each z"""

        M_of_zs = []
        for z in self.zs:
            M_of_zs.append(self.dkms_dMpc(z))

        return M_of_zs

    def get_new_model(self, zs, like_params):
        """For an arbitrary list of like_params, return a new CAMBModel"""

        # store a dictionary with parameters set to input values
        camb_param_dict = {}

        known_parameters = self.get_likelihood_parameters()
        for name, value in like_params.items():
            if name in known_parameters:
                camb_param_dict[name] = value

        # Preserve every unspecified fiducial parameter.
        new_params = dict(self.cosmo.input_cosmo_params_dict)
        new_params.update(camb_param_dict)
        new_cosmo = Cosmology(cosmo_params_dict=new_params)

        return CAMBModel(zs=zs, cosmo=new_cosmo)
