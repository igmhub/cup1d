import numpy as np
from lace.cosmo.cosmology import Cosmology
from cup1d.likelihood import parameter as likelihood_parameter


class CAMBModel(object):
    """Cache LaCE/CAMB cosmology calculations required by :class:`Theory`."""

    def __init__(
        self, zs, cosmo=None, z_star=3.0, kp_kms=0.009, fast_camb=True
    ):
        """Initialize a cosmology evaluator on a redshift grid.

        Parameters
        ----------
        zs : array_like
            Redshifts at which linear power is needed.
        cosmo : lace.cosmo.cosmology.Cosmology, optional
            Cosmology evaluator. A fiducial LaCE cosmology is created when
            omitted.
        z_star : float, default: 3.0
            Redshift for cached compressed linear-power parameters.
        kp_kms : float, default: 0.009
            Velocity-space pivot in ``s / km`` for compressed parameters.
        fast_camb : bool, default: True
            Compatibility flag retained by downstream callers.
        """

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
        """Construct sampled cosmological parameters and their top-hat bounds.

        Parameters
        ----------
        cosmo_priors : mapping, optional
            Optional ``As``, ``ns``, and ``nrun`` ``(minimum, maximum)``
            bounds replacing the built-in broad ranges.

        Returns
        -------
        dict of str to dict
            Parameter metadata for ``ombh2``, ``omch2``, ``As``, ``ns``,
            ``mnu``, ``nrun``, and ``H0``.
        """

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
        """Return cached or newly evaluated CAMB background results.

        Returns
        -------
        camb.results.CAMBdata
            Results supplied by the wrapped LaCE cosmology.
        """

        if self.cached_camb_results is None:
            self.cached_camb_results = self.cosmo.get_CAMBdata()

        return self.cached_camb_results

    def get_linP_Mpc(self):
        """Return cached linear power on the internal comoving-k grid.

        Returns
        -------
        tuple
            ``(k_Mpc, zs, linP_Mpc)`` with wavenumbers in ``1 / Mpc`` and
            power evaluated at the configured redshifts.
        """

        if self.cached_linP_Mpc is None:
            k_Mpc = np.logspace(-4, np.log10(self.cosmo.get_kmax_linP_Mpc()), 1000)
            linP_Mpc = self.cosmo.get_linP_Mpc(np.asarray(self.zs), k_Mpc)
            self.cached_linP_Mpc = (k_Mpc, list(self.zs), linP_Mpc)

        return self.cached_linP_Mpc

    def get_linP_params(self):
        """Return compressed linear-power parameters at the configured pivot.

        Returns
        -------
        dict
            Linear-power amplitude, slope, and running at ``z_star`` and
            ``kp_kms``.
        """

        if self.cached_linP_params is None:
            self.cached_linP_params = self.cosmo.get_linP_kms_params(
                self.z_star, self.kp_kms
            )

        return self.cached_linP_params

    def get_linP_Mpc_params(self, kp_Mpc):
        """Evaluate compressed comoving linear-power parameters at each redshift.

        Parameters
        ----------
        kp_Mpc : float
            Comoving pivot wavenumber in ``1 / Mpc``.

        Returns
        -------
        list of dict
            Amplitude, slope, and running dictionaries in configured-redshift
            order.
        """

        return [
            self.cosmo.get_linP_Mpc_params(z, kp_Mpc) for z in self.zs
        ]

    def dkms_dMpc(self, z):
        """Return the comoving-to-velocity wavenumber conversion at redshift.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            ``H(z) / (1 + z)`` in ``km / s / Mpc``.
        """

        return self.cosmo.get_dkms_dMpc(z)

    def get_M_of_zs(self):
        """Return comoving-to-velocity conversions for all configured redshifts.

        Returns
        -------
        list of float
            ``H(z) / (1 + z)`` values in ``km / s / Mpc``.
        """

        M_of_zs = []
        for z in self.zs:
            M_of_zs.append(self.dkms_dMpc(z))

        return M_of_zs

    def get_new_model(self, zs, like_params):
        """Create a new model after applying sampled cosmological parameters.

        Parameters
        ----------
        zs : array_like
            Redshifts for the new model.
        like_params : mapping
            Physical likelihood parameters. Recognized cosmology keys replace
            the corresponding values while unspecified inputs stay fiducial.

        Returns
        -------
        CAMBModel
            Fresh model with empty calculation caches.
        """

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
