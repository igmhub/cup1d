import numpy as np
from warnings import warn

from cup1d.p1ds.base_p1d_data import BaseDataP1D
from lace.utils.smoothing_manager import apply_smoothing


class BaseMockP1D(BaseDataP1D):
    """Base container for mock one-dimensional power-spectrum measurements."""

    def __init__(
        self,
        zs,
        k_kms,
        Pk_kms,
        cov_Pk_kms,
        full_zs=None,
        full_Pk_kms=None,
        full_cov_kms=None,
        full_cov_stat_kms=None,
        cov_stat=None,
        add_noise=False,
        seed=0,
        z_min=0,
        z_max=10,
        theory=None,
    ):
        """Initialize a mock P1D measurement, optionally adding Gaussian noise.

        Parameters
        ----------
        zs : array_like of float
            Redshift-bin centers.
        k_kms : sequence of ndarray
            Per-redshift wavenumber grids in ``s / km``.
        Pk_kms : sequence of ndarray
            Noise-free P1D measurements in ``km / s``.
        cov_Pk_kms : sequence of ndarray
            Per-redshift covariance blocks in ``(km / s)**2``.
        full_zs, full_Pk_kms, full_cov_kms, full_cov_stat_kms, cov_stat
            Optional concatenated and statistical-only data products forwarded
            to :class:`BaseDataP1D`.
        add_noise : bool, default: False
            Draw one correlated Gaussian realization in each redshift bin.
        seed : int, default: 0
            Seed used for the independent per-redshift random draws.
        z_min, z_max : float, default: 0, 10
            Inclusive redshift range retained by the base container.
        theory : object, optional
            Theory object used to store the cosmological and IGM truth values.
        """

        if add_noise:
            warn("Perturbing data by adding Gaussian noise")
            Pk_perturb_kms = self.get_Pk_iz_perturbed(Pk_kms, cov_Pk_kms, seed=seed)
        else:
            Pk_perturb_kms = Pk_kms

        if theory is not None:
            self.set_truth(theory, zs)

        super().__init__(
            zs,
            k_kms,
            Pk_perturb_kms,
            cov_Pk_kms,
            z_min=z_min,
            z_max=z_max,
            full_zs=full_zs,
            full_Pk_kms=full_Pk_kms,
            full_cov_kms=full_cov_kms,
            full_cov_stat_kms=full_cov_stat_kms,
            cov_stat=cov_stat,
        )

    def get_Pk_iz_perturbed(self, Pk_kms, cov_Pk_kms, nsamples=1, seed=0):
        """Draw correlated Gaussian P1D realizations independently by redshift.

        Parameters
        ----------
        Pk_kms : sequence of ndarray
            Mean P1D vectors in ``km / s``.
        cov_Pk_kms : sequence of ndarray
            Matching covariance blocks with shape ``(nk, nk)`` in
            ``(km / s)**2``.
        nsamples : int, default: 1
            Number of draws in every redshift bin.
        seed : int, default: 0
            Seed for NumPy's random generator.

        Returns
        -------
        list of ndarray
            One vector of shape ``(nk,)`` per redshift when ``nsamples == 1``;
            otherwise arrays of shape ``(nsamples, nk)``.  Cross-redshift
            covariance is not sampled.
        """

        rng = np.random.default_rng(seed)
        Pk_iz_perturb = []

        for iz in range(len(Pk_kms)):
            _ = rng.multivariate_normal(Pk_kms[iz], cov_Pk_kms[iz], nsamples)
            if nsamples == 1:
                Pk_iz_perturb.append(_[0])
            else:
                Pk_iz_perturb.append(_)

        return Pk_iz_perturb

    def set_smoothing_kms(self, emulator, fprint=print):
        """Smooth the stored P1D after converting it to comoving units.

        Parameters
        ----------
        emulator : object
            Emulator exposing the smoothing calibration accepted by
            :func:`lace.utils.smoothing_manager.apply_smoothing`.
        fprint : callable, default: print
            Status-reporting callback forwarded to the smoothing helper.

        Notes
        -----
        The method mutates :attr:`Pk_kms` in place.  It converts each
        wavenumber and P1D vector using the corresponding ``dkms_dMpc`` value.
        """

        list_data_Mpc = []
        for ii in range(len(self.z)):
            data = {}
            data["k_Mpc"] = self.k_kms * self.dkms_dMpc[ii]
            data["p1d_Mpc"] = self.Pk_kms[ii] * self.dkms_dMpc[ii]
            list_data_Mpc.append(data)

        apply_smoothing(emulator, list_data_Mpc, fprint=fprint)

        for ii in range(len(self.z)):
            self.Pk_kms[ii] = list_data_Mpc[ii]["p1d_Mpc_smooth"] / self.dkms_dMpc[ii]

    def set_smoothing_Mpc(self, emulator, list_data_Mpc, fprint=print):
        """Smooth externally supplied P1D arrays expressed in comoving units.

        Parameters
        ----------
        emulator : object
            Emulator defining the smoothing calibration.
        list_data_Mpc : list of dict
            One dictionary per redshift containing ``k_Mpc`` in ``1 / Mpc``
            and ``p1d_Mpc`` in ``Mpc``.
        fprint : callable, default: print
            Status-reporting callback forwarded to the smoothing helper.

        Returns
        -------
        list of dict
            Input dictionaries, with ``p1d_Mpc`` replaced by
            ``p1d_Mpc_smooth`` where the helper provided a smoothed result.
        """

        apply_smoothing(emulator, list_data_Mpc, fprint=fprint)
        print(list_data_Mpc[0]["k_Mpc"].max())
        for ii in range(len(list_data_Mpc)):
            if "p1d_Mpc_smooth" in list_data_Mpc[ii]:
                list_data_Mpc[ii]["p1d_Mpc"] = list_data_Mpc[ii]["p1d_Mpc_smooth"]

        return list_data_Mpc

    def plot_igm(self):
        """Plot the truth IGM history stored for this mock.

        Returns
        -------
        object
            Figure or axes object returned by
            :func:`cup1d.postprocessing.igm.plot_mock_igm`.
        """
        from cup1d.postprocessing.igm import plot_mock_igm as _plot

        return _plot(self)

    def set_truth(self, theory, zs):
        """Record cosmological, linear-power, and IGM truth values.

        Parameters
        ----------
        theory : object
            Cup1D theory object with initialized fiducial cosmology and IGM
            models.
        zs : array_like of float
            Redshifts at which to evaluate the IGM history.

        Notes
        -----
        The method replaces :attr:`truth` with ``cosmo``, ``linP``, and ``igm``
        mappings.  Thermal velocity and pressure scales are recorded in
        ``km / s`` and ``s / km``, respectively.
        """
        # setup fiducial cosmology
        self.truth = {}

        sim_cosmo = theory.fid_cosmo["cosmo"]
        background = sim_cosmo.get_background_params()
        primordial = sim_cosmo.get_primordial_params()

        self.truth["cosmo"] = {
            "ombh2": background["ombh2"],
            "omch2": background["omch2"],
            "As": primordial["As"],
            "ns": primordial["ns"],
            "nrun": primordial["nrun"],
            "H0": sim_cosmo.get_H0(),
            "mnu": sim_cosmo.get_mnu(),
        }

        self.truth["linP"] = {}
        cosmo_params = ["Delta2_star", "n_star", "alpha_star"]
        for par in cosmo_params:
            self.truth["linP"][par] = theory.fid_cosmo["linP_params"][par]

        truth_z = np.asarray(zs)
        self.truth["igm"] = {
            "z": truth_z,
            "tau_eff": theory.model_igm.models["F_model"].get_tau_eff(
                truth_z
            ),
            "gamma": theory.model_igm.models["T_model"].get_gamma(truth_z),
            "sigT_kms": theory.model_igm.models["T_model"].get_sigT_kms(
                truth_z
            ),
            "kF_kms": theory.model_igm.models["P_model"].get_kF_kms(truth_z),
        }
        # self.truth["cont"] = theory.model_cont.get_dict_cont()

    # def _get_cosmo(self, nyx_version="Jul2024"):
    #     # get cosmology
    #     fname = get_nyx_path() / ("nyx_emu_cosmo_" + nyx_version + ".npy")
    #     data_cosmo = np.load(fname, allow_pickle=True)

    #     true_cosmo = None
    #     for ii in range(len(data_cosmo)):
    #         if data_cosmo[ii]["sim_label"] == self.input_sim:
    #             true_cosmo = camb_cosmo.get_Nyx_cosmology(
    #                 data_cosmo[ii]["cosmo_params"]
    #             )
    #             break
    #     if true_cosmo is None:
    #         raise ValueError(f"Cosmo not found in {fname} for {self.input_sim}")

    #     return true_cosmo

    # def _get_igm(self):
    #     """Load IGM history"""
    #     fname = get_nyx_path() / "IGM_histories.npy"
    #     igm_hist = np.load(fname, allow_pickle=True).item()
    #     if self.input_sim not in igm_hist:
    #         raise ValueError(
    #             self.input_sim
    #             + " not found in "
    #             + fname
    #             + r"\n Check out the LaCE script save_"
    #             + self.input_sim[:3]
    #             + "_IGM.py"
    #         )
    #     else:
    #         true_igm = igm_hist[self.input_sim]

    #     return true_igm
