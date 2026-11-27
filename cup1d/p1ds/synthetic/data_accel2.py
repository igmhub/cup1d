import os

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

from cup1d.theory import cosmology as cosmologies
from cup1d.p1ds.synthetic.base_p1d_mock import BaseMockP1D
from cup1d.models.observation import apply_observation_model
from cup1d.p1ds.observations import (
    data_PD2013,
    data_Chabanier2019,
    data_Karacayli2022,
    data_DESIY1,
)


def load_data(folder, sim_label="l160_r25", hh=0.675, kmax=10):
    """Load P1D and angle-averaged P3D products from an ACCEL2 simulation.

    Parameters
    ----------
    folder : str or path-like
        Root directory containing ``p1d_from_sim`` and ``p3d_from_sim``.
    sim_label : str, default: "l160_r25"
        ACCEL2 simulation directory name.
    hh : float, default: 0.675
        Dimensionless Hubble parameter used to convert the stored ``h`` units.
    kmax : float, default: 10
        Strict upper comoving wavenumber cut in ``1 / Mpc``.

    Returns
    -------
    dict
        Redshifts; P1D grids, directional measurements, and mode counts; and
        P3D ``k``--``mu`` grids, mode counts, and powers.  P1D is in ``Mpc``;
        P3D is in ``Mpc**3``; wavenumbers are in ``1 / Mpc``.
    """

    labs_dirs = ["x", "y", "z"]
    labs_z = [
        "plt02937_2.0_",
        "plt02122_2.6_",
        "plt01902_3.0_",
        "plt01623_3.6_",
        "plt01392_4.0_",
        "plt00908_5.0_",
    ]
    zz = np.array([2.0, 2.6, 3.0, 3.6, 4.0, 5.0])
    folder1d = os.path.join(folder, "p1d_from_sim", sim_label)
    folder3d = os.path.join(folder, "p3d_from_sim", sim_label)

    for iz, labz in enumerate(labs_z):
        for ii, ax in enumerate(labs_dirs):
            file = labz + "f" + ax + "_flux_ps1d.txt"
            dat = np.loadtxt(os.path.join(folder1d, file))
            _ = dat[:, 0] < kmax
            if (ii == 0) & (iz == 0):
                k1d = dat[_, 0] * hh  # the beginning of the bin!
                # defined at the beginning of the bin, now center (equal bins)
                k1d += 0.5 * (k1d[1] - k1d[0])

                nmod1d = dat[_, 1]
                p1d_all = np.zeros((len(labs_z), k1d.shape[0], 3))
                p1d = np.zeros((len(labs_z), k1d.shape[0]))

                p1d[iz] = dat[_, 3] / hh
                p1d_all[iz, :, ii] = p1d[iz]
            else:
                _p1d = dat[_, 3] / hh
                p1d[iz] += _p1d
                p1d_all[iz, :, ii] = _p1d
        p1d[iz] /= len(labs_dirs)

        file = labz + "fmeandirection_flux_pkmu.txt"
        dat = np.loadtxt(os.path.join(folder3d, file))
        _ = dat[:, 0] < kmax

        if iz == 0:
            u_k3d = dat[_, 0] * hh  # the beginning of the bin!
            u_mu3d = dat[_, 1]  # the beginning of the bin!
            u_nmod3d = dat[_, 2]

            k3d_uni = np.unique(u_k3d)
            mu3d_uni = np.unique(u_mu3d)

            # defined at the beginning of the bin, now center (equal bins)
            diffk = 0.5 * (k3d_uni[1] - k3d_uni[0])
            diffmu = 0.5 * (mu3d_uni[1] - mu3d_uni[0])

            u_k3d += diffk
            k3d_uni += diffk
            u_mu3d += diffmu
            mu3d_uni += diffmu

            k3d = np.zeros((k3d_uni.shape[0], mu3d_uni.shape[0]))
            mu3d = np.zeros_like(k3d)
            nmod3d = np.zeros_like(k3d)
            p3d = np.zeros((len(labs_z), k3d_uni.shape[0], mu3d_uni.shape[0]))

        u_p3d = dat[_, 5] / hh**3

        for ii in range(k3d.shape[0]):
            for jj in range(k3d.shape[1]):
                _ = np.argwhere((u_k3d == k3d_uni[ii]) & (u_mu3d == mu3d_uni[jj]))[0, 0]
                if iz == 0:
                    k3d[ii, jj] = u_k3d[_]
                    mu3d[ii, jj] = u_mu3d[_]
                    nmod3d[ii, jj] = u_nmod3d[_]
                p3d[iz, ii, jj] = u_p3d[_]

        dict_out = {
            "z": zz,
            "k1d_Mpc": k1d,
            "p1d_Mpc": p1d,
            "p1d_Mpc_axes": p1d_all,
            "nmod1d": nmod1d,
            "k3d_Mpc": k3d,
            "mu3d": mu3d,
            "nmod3d": nmod3d,
            "p3d_Mpc": p3d,
        }

    return dict_out


class Accel2_P1D(BaseMockP1D):
    """Load an ACCEL2 MP-Gadget simulation as a mock P1D data object.

    The mock can inherit PD2013 or Chabanier2019 observational covariance
    layouts while retaining the simulation's underlying P1D signal.
    """

    def __init__(
        self,
        theory,
        testing_data=None,
        apply_smoothing=True,
        input_sim="l160_r25",
        data_cov_label="Chabanier2019",
        add_syst=True,
        add_noise=False,
        seed=0,
        z_min=0,
        z_max=10,
        path_data=None,
        p1d_fname=None,
        interp_to_cov=False,
    ):
        """Build an ACCEL2 mock on a selected observational covariance layout.

        Parameters
        ----------
        theory : object
            Initialized Cup1D theory, used for coordinate conversion,
            smoothing, contaminants, and truth metadata.
        testing_data : object, optional
            Reserved compatibility argument; ACCEL2 products are loaded from
            ``path_data``.
        apply_smoothing : bool, default: True
            Apply the emulator's calibrated smoothing to simulation P1D.
        input_sim : str, default: "l160_r25"
            ACCEL2 simulation label.
        data_cov_label : str, default: "Chabanier2019"
            Observational product providing covariance and target sampling.
        add_syst, add_noise : bool, default: True, False
            Include reference systematic covariance and optionally realize
            Gaussian measurement noise.
        seed : int, default: 0
            Seed used when drawing mock noise.
        z_min, z_max : float, default: 0, 10
            Inclusive output-redshift interval.
        path_data : str or path-like
            ACCEL2 product root directory.
        p1d_fname : str or path-like, optional
            Explicit DESI reference P1D product when relevant.
        interp_to_cov : bool, default: False
            If true, interpolate simulation P1D to covariance grids; otherwise
            interpolate covariance sampling to the native simulation grids.
        """

        # covariance matrix settings
        self.add_syst = add_syst
        self.data_cov_label = data_cov_label
        self.input_sim = input_sim

        # load data from simulations
        out_dict = load_data(path_data)
        sim_cosmo = cosmologies.set_cosmo(cosmo_label="accel2")

        testing_data = []
        for ii in range(out_dict["p1d_Mpc"].shape[0]):
            # flux rescaled to this value in ACCEL2
            tau_eff = 0.0025 * (1 + out_dict["z"][ii]) ** 3.7
            testing_data.append(
                {
                    "z": out_dict["z"][ii],
                    "k_Mpc": out_dict["k1d_Mpc"],
                    "p1d_Mpc": out_dict["p1d_Mpc"][ii],
                    "dkms_dMpc": sim_cosmo.get_dkms_diMpc(out_dict["z"][ii]),
                    "mF": np.exp(-tau_eff),
                }
            )

        if apply_smoothing:
            self.testing_data = super().set_smoothing_Mpc(theory.emulator, testing_data)
        else:
            print("No smoothing is applied")
            self.testing_data = testing_data

        # store cosmology used in the simulation
        dkms_dMpc = []
        for ii in range(len(testing_data)):
            dkms_dMpc.append(testing_data[ii]["dkms_dMpc"])
        self.dkms_dMpc = np.array(dkms_dMpc)

        if interp_to_cov:
            prepare_mock = self._load_p1d_to_cov
        else:
            prepare_mock = self._load_p1d
        (
            zs,
            k_kms,
            Pk_kms,
            cov,
            cov_stat,
            full_zs,
            full_Pk_kms,
            full_cov_kms,
            full_cov_stat_kms,
        ) = prepare_mock(theory, p1d_fname=p1d_fname)

        # set theory (just to save truth)
        zs = np.array(zs)
        theory.model_igm.set_fid_igm(zs)
        theory.set_fid_cosmo(zs)

        # apply the observation model at the fiducial nuisance values, as the
        # theory does: (C_HCD C_mul IC P + C_add) C_res
        mF = theory.model_igm.models["F_model"].get_mean_flux(zs)
        # one value per data redshift (fid_cosmo["M_of_zs"] also has z_star)
        M_of_z = theory.fid_cosmo["cosmo"].get_dkms_dMpc(zs)
        Pk_kms, _ = apply_observation_model(
            theory.model_cont, theory.model_syst, zs, k_kms, Pk_kms, mF, M_of_z
        )
        full_Pk_kms = np.concatenate(Pk_kms)

        # setup base class
        super().__init__(
            zs,
            k_kms,
            Pk_kms,
            cov,
            full_zs=full_zs,
            full_Pk_kms=full_Pk_kms,
            full_cov_kms=full_cov_kms,
            full_cov_stat_kms=full_cov_stat_kms,
            cov_stat=cov_stat,
            add_noise=add_noise,
            seed=seed,
            z_min=z_min,
            z_max=z_max,
            theory=theory,
        )

        return

    # def set_truth(self, theory, zs):
    #     # setup fiducial cosmology
    #     self.truth = {}

    #     sim_cosmo = theory.fid_cosmo["cosmo"].cosmo

    #     self.truth["cosmo"] = {}
    #     self.truth["cosmo"]["ombh2"] = sim_cosmo.ombh2
    #     self.truth["cosmo"]["omch2"] = sim_cosmo.omch2
    #     self.truth["cosmo"]["As"] = sim_cosmo.InitPower.As
    #     self.truth["cosmo"]["ns"] = sim_cosmo.InitPower.ns
    #     self.truth["cosmo"]["nrun"] = sim_cosmo.InitPower.nrun
    #     self.truth["cosmo"]["H0"] = sim_cosmo.H0
    #     self.truth["cosmo"]["mnu"] = camb_cosmo.get_mnu(sim_cosmo)

    #     self.truth["linP"] = {}
    #     blob_params = ["Delta2_star", "n_star", "alpha_star"]
    #     blob = theory.fid_cosmo["cosmo"].get_linP_params()
    #     for ii in range(len(blob_params)):
    #         self.truth["linP"][blob_params[ii]] = blob[blob_params[ii]]

    #     self.truth["igm"] = {}
    #     self.truth["igm"]["label"] = self.input_sim
    #     zs = np.array(zs)
    #     self.truth["igm"]["z"] = zs
    #     self.truth["igm"]["tau_eff"] = theory.model_igm.F_model.get_tau_eff(zs)
    #     self.truth["igm"]["gamma"] = theory.model_igm.T_model.get_gamma(zs)
    #     self.truth["igm"]["sigT_kms"] = theory.model_igm.T_model.get_sigT_kms(
    #         zs
    #     )
    #     self.truth["igm"]["kF_kms"] = theory.model_igm.P_model.get_kF_kms(zs)

    #     self.truth["cont"] = {}
    #     for ii in range(2):
    #         self.truth["cont"][
    #             "ln_SiIII_" + str(ii)
    #         ] = theory.model_cont.fid_SiIII[-1 - ii][-1]
    #         self.truth["cont"][
    #             "d_SiIII_" + str(ii)
    #         ] = theory.model_cont.fid_SiIII[-1 - ii][0]
    #         self.truth["cont"][
    #             "ln_SiII_" + str(ii)
    #         ] = theory.model_cont.fid_SiII[-1 - ii][-1]
    #         self.truth["cont"][
    #             "d_SiII_" + str(ii)
    #         ] = theory.model_cont.fid_SiII[-1 - ii][0]
    #         self.truth["cont"][
    #             "ln_A_damp_" + str(ii)
    #         ] = theory.model_cont.fid_HCD[-1 - ii]
    #         self.truth["cont"]["ln_SN_" + str(ii)] = theory.model_cont.fid_SN[
    #             -1 - ii
    #         ]
    #         self.truth["cont"]["ln_AGN_" + str(ii)] = theory.model_cont.fid_AGN[
    #             -1 - ii
    #         ]

    def _load_p1d(self, theory, p1d_fname=None):
        """Interpolate ACCEL2 P1D onto the covariance reference layout.

        Parameters
        ----------
        theory : object
            Theory used to obtain redshift-dependent velocity--comoving
            conversion factors.
        p1d_fname : str or path-like, optional
            Explicit DESI reference data file when ``data_cov_label`` selects
            a DESI product.

        Returns
        -------
        tuple
            Per-redshift P1D in ``km / s``, wavenumbers in ``s / km``, total
            and statistical covariance blocks in ``(km / s)**2``, and their
            concatenated representations.
        """
        # figure out dataset to mimic
        if self.data_cov_label == "Chabanier2019":
            data = data_Chabanier2019.P1D_Chabanier2019(add_syst=self.add_syst)
        elif self.data_cov_label == "PD2013":
            data = data_PD2013.P1D_PD2013(add_syst=self.add_syst)
        elif self.data_cov_label == "Karacayli2022":
            data = data_Karacayli2022.P1D_Karacayli2022()
        elif self.data_cov_label.startswith("DESIY1"):
            data = data_DESIY1.P1D_DESIY1(
                data_label=self.data_cov_label, p1d_fname=p1d_fname
            )
        else:
            raise ValueError("Unknown data_cov_label", self.data_cov_label)

        # set interpolator
        z_sim = []
        k_Mpc_sim = []
        p1d_Mpc_sim = []
        for sim in self.testing_data:
            z_sim.append(sim["z"])
            # select only data within 10 Mpc
            _ = (sim["k_Mpc"] > 0) & (sim["k_Mpc"] < 10)
            k_Mpc_sim.append(sim["k_Mpc"][_])
            p1d_Mpc_sim.append(sim["p1d_Mpc"][_])
        z_sim = np.array(z_sim)
        k_Mpc_sim = np.array(k_Mpc_sim)
        p1d_Mpc_sim = np.array(p1d_Mpc_sim)

        theory.set_fid_cosmo(z_sim)

        interp = RegularGridInterpolator((z_sim, k_Mpc_sim[0]), np.log(p1d_Mpc_sim))

        zs = data.z
        k_kms = []
        Pk_kms = []
        cov = []
        cov_stat = []
        full_zs = []
        for ii, _k_kms in enumerate(data.k_kms):
            dkms_dMpc = theory.fid_cosmo["cosmo"].get_dkms_dMpc(zs[ii])
            # convert Mpc to km/s
            data_k_Mpc = _k_kms * dkms_dMpc
            # cutting scales too large for the simulation
            _ = data_k_Mpc >= k_Mpc_sim[0].min()
            Pk_Mpc = np.exp(interp((zs[ii], data_k_Mpc[_])))
            # convert Mpc to km/s
            k_kms.append(_k_kms[_])
            Pk_kms.append(Pk_Mpc * dkms_dMpc)
            cov.append(data.cov_Pk_kms[ii][_][:, _])
            cov_stat.append(data.covstat_Pk_kms[ii][_][:, _])
            full_zs.append(zs[ii] * np.ones(len(_k_kms[_])))
        full_zs = np.concatenate(full_zs)
        full_k_kms = np.concatenate(k_kms)
        full_Pk_kms = np.concatenate(Pk_kms)

        # remove scales not used from cov matrix
        full_cov_kms = np.zeros((len(full_Pk_kms), len(full_Pk_kms)))
        full_cov_stat_kms = np.zeros((len(full_Pk_kms), len(full_Pk_kms)))
        for i0 in range(len(full_Pk_kms)):
            ind0 = np.argwhere(full_zs[i0] == data.full_zs)[:, 0]
            j0 = ind0[np.argmin(abs(full_k_kms[i0] - data.full_k_kms[ind0]))]
            for i1 in range(len(full_Pk_kms)):
                ind1 = np.argwhere(full_zs[i1] == data.full_zs)[:, 0]
                j1 = ind1[np.argmin(abs(full_k_kms[i1] - data.full_k_kms[ind1]))]
                full_cov_kms[i0, i1] = data.full_cov_Pk_kms[j0, j1]
                full_cov_stat_kms[i0, i1] = data.full_cov_stat_Pk_kms[j0, j1]

        return (
            zs,
            k_kms,
            Pk_kms,
            cov,
            cov_stat,
            full_zs,
            full_Pk_kms,
            full_cov_kms,
            full_cov_stat_kms,
        )

    def _load_p1d_to_cov(self, theory, p1d_fname=None):
        """Map covariance sampling to the native ACCEL2 redshift grid.

        Parameters
        ----------
        theory : object
            Theory used for comoving--velocity conversion.
        p1d_fname : str or path-like, optional
            Explicit DESI reference data file when required.

        Returns
        -------
        tuple
            Native ACCEL2 redshifts, P1D vectors in ``km / s``, wavenumbers in
            ``s / km``, total and statistical covariance blocks in
            ``(km / s)**2``, and concatenated representations.
        """
        # figure out dataset to mimic
        if self.data_cov_label == "Chabanier2019":
            data = data_Chabanier2019.P1D_Chabanier2019(add_syst=self.add_syst)
        elif self.data_cov_label == "PD2013":
            data = data_PD2013.P1D_PD2013(add_syst=self.add_syst)
        elif self.data_cov_label == "Karacayli2022":
            data = data_Karacayli2022.P1D_Karacayli2022()
        elif self.data_cov_label.startswith("DESIY1"):
            data = data_DESIY1.P1D_DESIY1(
                data_label=self.data_cov_label, p1d_fname=p1d_fname
            )
        else:
            raise ValueError("Unknown data_cov_label", self.data_cov_label)

        # get redshifts in testing simulation
        z_sim = np.array([data["z"] for data in self.testing_data])

        # unit conversion, at zmin to get lowest possible k_min_kms
        dkms_dMpc_zmin = self.dkms_dMpc[np.argmin(z_sim)]

        # Get k_min for the sim data & cut k values below that
        k_min_Mpc = self.testing_data[0]["k_Mpc"][0]
        if k_min_Mpc == 0:
            k_min_Mpc = self.testing_data[0]["k_Mpc"][1]
        k_min_kms = k_min_Mpc / dkms_dMpc_zmin

        k_kms = []
        Pk_kms = []
        cov = []
        cov_stat = []
        zs_native = []
        zs_cov = []

        for iz in range(len(z_sim)):
            z = z_sim[iz]
            iz_data = np.argmin(np.abs(data.z - z))
            # print(z, data.z[iz_data])

            # interpolate native k to cov k
            Ncull = np.sum(data.k_kms[iz_data] < k_min_kms)
            _k_kms = data.k_kms[iz_data][Ncull:]
            k_kms.append(_k_kms)

            # convert Mpc to km/s
            data_k_Mpc = np.array(_k_kms) * self.dkms_dMpc[iz]

            # find testing data for this redshift
            sim_k_Mpc = self.testing_data[iz]["k_Mpc"].copy()
            sim_p1d_Mpc = self.testing_data[iz]["p1d_Mpc"].copy()

            # mask k=0 if present
            if sim_k_Mpc[0] == 0:
                sim_k_Mpc = sim_k_Mpc[1:]
                sim_p1d_Mpc = sim_p1d_Mpc[1:]

            interp_sim_Mpc = interp1d(sim_k_Mpc, sim_p1d_Mpc, "cubic")
            sim_p1d_kms = interp_sim_Mpc(data_k_Mpc) * self.dkms_dMpc[iz]

            # append redshift, p1d and covar
            zs_native.append(z)
            zs_cov.append(data.z[iz_data])
            Pk_kms.append(sim_p1d_kms)
            # # Now get covariance from the nearest z bin in data
            cov.append(data.cov_Pk_kms[iz_data][Ncull:, Ncull:])
            cov_stat.append(data.covstat_Pk_kms[iz_data][Ncull:, Ncull:])

        full_k_kms = np.concatenate(k_kms)
        full_zs_native = []
        full_zs_cov = []
        for ii in range(len(k_kms)):
            full_zs_native.append(np.ones(len(k_kms[ii])) * zs_native[ii])
            full_zs_cov.append(np.ones(len(k_kms[ii])) * zs_cov[ii])
        full_zs_native = np.concatenate(full_zs_native)
        full_zs_cov = np.concatenate(full_zs_cov)
        full_Pk_kms = np.concatenate(Pk_kms)

        full_cov_kms = np.zeros((len(full_Pk_kms), len(full_Pk_kms)))
        full_cov_stat_kms = np.zeros((len(full_Pk_kms), len(full_Pk_kms)))
        for i0 in range(len(full_Pk_kms)):
            ind0 = np.argwhere(full_zs_cov[i0] == data.full_zs)[:, 0]
            j0 = ind0[np.argmin(abs(full_k_kms[i0] - data.full_k_kms[ind0]))]
            for i1 in range(len(full_Pk_kms)):
                ind1 = np.argwhere(full_zs_cov[i1] == data.full_zs)[:, 0]
                j1 = ind1[np.argmin(abs(full_k_kms[i1] - data.full_k_kms[ind1]))]
                full_cov_kms[i0, i1] = data.full_cov_Pk_kms[j0, j1]
                full_cov_stat_kms[i0, i1] = data.full_cov_stat_Pk_kms[j0, j1]

        return (
            zs_native,
            k_kms,
            Pk_kms,
            cov,
            cov_stat,
            full_zs_native,
            full_Pk_kms,
            full_cov_kms,
            full_cov_stat_kms,
        )
