"""CAMB-backed cosmology model used by the Lyman-alpha theory layer."""

from __future__ import annotations

from typing import Any

import numpy as np
from lace.cosmo import camb_cosmo, fit_linP

from cup1d.likelihood import likelihood_parameter


class CAMBModel:
    """Interface between a CAMB cosmology object and :class:`Theory`.

    Parameters
    ----------
    zs : np.ndarray
        List of redshifts at which we evaluate linear power.
    cosmo : Any, optional
        CAMB cosmology object. If None, a default cosmology is used.
    z_star : float, optional
        Pivot redshift for linear power parameters. Default is 3.0.
    kp_kms : float, optional
        Pivot wavenumber in km/s. Default is 0.009.
    fast_camb : bool, optional
        Whether to use fast CAMB evaluation. Default is True.

    Attributes
    ----------
    zs : np.ndarray
        Redshifts for evaluation.
    cosmo : Any
        CAMB cosmology object.
    z_star : float
        Pivot redshift.
    kp_kms : float
        Pivot wavenumber.
    fast_camb : bool
        Fast CAMB flag.
    cached_camb_results : Any
        Cached CAMB results object.
    cached_linP_Mpc : tuple[np.ndarray, np.ndarray, np.ndarray]
        Cached linear power in Mpc.
    cached_linP_params : dict[str, float]
        Cached linear power parameters.
    """

    def __init__(
        self,
        zs: np.ndarray,
        cosmo: Any | None = None,
        z_star: float = 3.0,
        kp_kms: float = 0.009,
        fast_camb: bool = True,
    ):
        """Initialize the CAMB model."""

        # list of redshifts at which we evaluate linear power
        self.zs = zs
        self.fast_camb = fast_camb

        # setup CAMB cosmology object
        if cosmo is None:
            self.cosmo = camb_cosmo.get_cosmology()
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

    def get_likelihood_parameters(
        self, cosmo_priors: dict | None = None
    ) -> list[likelihood_parameter.LikelihoodParameter]:
        """Return cosmological likelihood parameters.

        Parameters
        ----------
        cosmo_priors : dict, optional
            Dictionary of cosmological priors.

        Returns
        -------
        list[likelihood_parameter.LikelihoodParameter]
            List of likelihood parameters.
        """

        # should clarify role of min/max given that these are also
        # set in the likelihood

        params = []
        params.append(
            likelihood_parameter.LikelihoodParameter(
                name="ombh2",
                min_value=0.018,
                max_value=0.026,
                value=self.cosmo.ombh2,
            )
        )
        params.append(
            likelihood_parameter.LikelihoodParameter(
                name="omch2",
                min_value=0.10,
                max_value=0.14,
                value=self.cosmo.omch2,
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
            likelihood_parameter.LikelihoodParameter(
                name="As",
                min_value=min_val,
                max_value=max_val,
                value=self.cosmo.InitPower.As,
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
            likelihood_parameter.LikelihoodParameter(
                name="ns",
                min_value=min_val,
                max_value=max_val,
                value=self.cosmo.InitPower.ns,
            )
        )
        params.append(
            likelihood_parameter.LikelihoodParameter(
                name="mnu",
                min_value=0.0,
                max_value=1.0,
                value=camb_cosmo.get_mnu(self.cosmo),
            )
        )

        if cosmo_priors is not None:
            min_val = cosmo_priors["nrun"][0]
            max_val = cosmo_priors["nrun"][1]
        else:
            min_val = -0.05
            max_val = 0.05
        params.append(
            likelihood_parameter.LikelihoodParameter(
                name="nrun",
                min_value=min_val,
                max_value=max_val,
                value=self.cosmo.InitPower.nrun,
            )
        )
        params.append(
            likelihood_parameter.LikelihoodParameter(
                name="H0", min_value=50, max_value=100, value=self.cosmo.H0
            )
        )

        return params

    def get_camb_results(self) -> Any:
        """Return cached CAMB results, computing them if needed.

        Returns
        -------
        Any
            CAMB results object.
        """

        if self.cached_camb_results is None:
            self.cached_camb_results = camb_cosmo.get_camb_results(
                self.cosmo, zs=self.zs, fast_camb=self.fast_camb
            )

        return self.cached_camb_results

    def get_linP_Mpc(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached ``(k_Mpc, zs, linP_Mpc)`` arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Wavenumbers, redshifts, and linear power in Mpc.
        """

        if self.cached_linP_Mpc is None:
            camb_results = self.get_camb_results()
            self.cached_linP_Mpc = camb_cosmo.get_linP_Mpc(
                pars=self.cosmo, zs=self.zs, camb_results=camb_results
            )

        return self.cached_linP_Mpc

    def get_linP_params(self) -> dict[str, float]:
        """Return linear-power parameters at ``(z_star, kp_kms)``.

        Returns
        -------
        dict[str, float]
            Dictionary of linear power parameters.
        """

        if self.cached_linP_params is None:
            self.cached_linP_params = fit_linP.parameterize_cosmology_kms(
                self.cosmo,
                self.get_camb_results(),
                self.z_star,
                self.kp_kms,
                fast_camb=self.fast_camb,
            )

        return self.cached_linP_params

    def get_linP_Mpc_params(self, kp_Mpc: float) -> list[dict[str, float]]:
        """Return emulator linear-power parameters around ``kp_Mpc``.

        Parameters
        ----------
        kp_Mpc : float
            Pivot wavenumber in Mpc.

        Returns
        -------
        list[dict[str, float]]
            List of linear power parameters for each redshift.
        """

        ## Get the P(k) at each z
        k_Mpc, z, pk_Mpc = self.get_linP_Mpc()

        # specify wavenumber range to fit
        kmin_Mpc = 0.5 * kp_Mpc
        kmax_Mpc = 2.0 * kp_Mpc

        linP_params = []
        ## Fit the emulator call params
        for pk_z in pk_Mpc:
            linP_Mpc = fit_linP.fit_polynomial(
                kmin_Mpc / kp_Mpc,
                kmax_Mpc / kp_Mpc,
                k_Mpc / kp_Mpc,
                pk_z,
                deg=2,
            )
            # translate the polynomial to our parameters
            ln_A_p = linP_Mpc[0]
            Delta2_p = np.exp(ln_A_p) * kp_Mpc**3 / (2 * np.pi**2)
            n_p = linP_Mpc[1]
            # note that the curvature is alpha/2
            alpha_p = 2.0 * linP_Mpc[2]
            linP_z = {"Delta2_p": Delta2_p, "n_p": n_p, "alpha_p": alpha_p}
            linP_params.append(linP_z)

        return linP_params

    def dkms_dMpc(self, z: float) -> float:
        """Return ``H(z)/(1+z)`` to convert Mpc to km/s.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            Conversion factor.
        """

        # get CAMB results objects (might be cached already)
        camb_results = self.get_camb_results()
        H_z = camb_results.hubble_parameter(z)
        return H_z / (1 + z)

    def get_M_of_zs(self) -> list[float]:
        """Return ``M(z)=H(z)/(1+z)`` for every model redshift.

        Returns
        -------
        list[float]
            List of conversion factors for each model redshift.
        """

        M_of_zs = []
        for z in self.zs:
            M_of_zs.append(self.dkms_dMpc(z))

        return M_of_zs

    def get_new_model(
        self,
        zs: np.ndarray,
        like_params: list[likelihood_parameter.LikelihoodParameter],
    ) -> CAMBModel:
        """Return a new :class:`CAMBModel` updated from likelihood parameters.

        Parameters
        ----------
        zs : np.ndarray
            Redshifts for the new model.
        like_params : list[likelihood_parameter.LikelihoodParameter]
            List of likelihood parameters.

        Returns
        -------
        CAMBModel
            New CAMB model.
        """

        # store a dictionary with parameters set to input values
        camb_param_dict = {}

        # loop over list of likelihood parameters own by this object
        for mypar in self.get_likelihood_parameters():
            # loop over input parameters
            for inpar in like_params:
                if inpar.name == mypar.name:
                    camb_param_dict[inpar.name] = inpar.value
                    continue

        # set cosmology object (use fiducial for parameters not provided)
        new_cosmo = camb_cosmo.get_cosmology_from_dictionary(
            camb_param_dict, cosmo_fid=self.cosmo
        )

        return CAMBModel(zs=zs, cosmo=new_cosmo)
