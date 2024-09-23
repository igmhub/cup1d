import os
import lace
import numpy as np

from cup1d.nuisance import mean_flux_model, thermal_model, pressure_model


class IGM(object):
    """Contains all IGM models"""

    def __init__(
        self,
        zs,
        free_param_names=None,
        z_pivot=3,
        F_model=None,
        T_model=None,
        P_model=None,
        fid_sim_igm="mpg_central",
        list_sim_cube=None,
        type_priors="hc",
    ):
        # load fiducial IGM history (used for fitting)
        self.fid_sim_igm = fid_sim_igm
        self.z_pivot = z_pivot

        fid_igm = self.get_igm(fid_sim_igm)

        # compute priors for this emulator
        if list_sim_cube is None:
            # default priors (hc for mpg)
            self.priors = {
                "tau_eff": [
                    [-0.5912022429177898, 0.5912022429177898],
                    [-0.1936758618341875, 0.20169231438172694],
                ],
                "gamma": [
                    [-0.6257448015637526, 0.6257448015637526],
                    [-0.2260619891767361, 0.19240662107387235],
                ],
                "sigT_kms": [
                    [-0.5683643384807968, 0.5683643384807968],
                    [-0.18454429411238835, 0.19555096875785932],
                ],
                "kF_kms": [
                    [-0.5470878025948258, 0.5470878025948258],
                    [-0.15974048204662275, 0.2061260371234787],
                ],
            }
        else:
            self.set_priors(fid_igm, list_sim_cube, type_priors=type_priors)

        # setup fiducial IGM models
        if F_model is not None:
            self.F_model = F_model
        else:
            self.F_model = mean_flux_model.MeanFluxModel(
                free_param_names=free_param_names,
                fid_igm=fid_igm,
                z_tau=z_pivot,
                priors=self.priors,
            )
        if T_model:
            self.T_model = T_model
        else:
            self.T_model = thermal_model.ThermalModel(
                free_param_names=free_param_names,
                fid_igm=fid_igm,
                z_T=z_pivot,
                priors=self.priors,
            )
        if P_model:
            self.P_model = P_model
        else:
            self.P_model = pressure_model.PressureModel(
                free_param_names=free_param_names,
                fid_igm=fid_igm,
                z_kF=z_pivot,
                priors=self.priors,
            )

        self.fid_igm = {}
        self.fid_igm["z"] = zs
        self.fid_igm["tau_eff"] = self.F_model.get_tau_eff(zs)
        self.fid_igm["gamma"] = self.T_model.get_gamma(zs)
        self.fid_igm["sigT_kms"] = self.T_model.get_sigT_kms(zs)
        self.fid_igm["kF_kms"] = self.P_model.get_kF_kms(zs)

    def get_igm(self, sim_igm, return_all=False):
        """Load IGM history"""
        if sim_igm[:3] == "mpg":
            repo = os.path.dirname(lace.__path__[0]) + "/"
            fname = repo + "/data/sim_suites/Australia20/IGM_histories.npy"
        elif sim_igm[:3] == "nyx":
            fname = os.environ["NYX_PATH"] + "/IGM_histories.npy"
        else:
            raise ValueError("only mpg and nyx sim_igm implemented")

        try:
            igm_hist = np.load(fname, allow_pickle=True).item()
        except:
            raise ValueError(
                fname
                + " not found. You can produce it using LaCE"
                + r" script save_"
                + sim_igm[:3]
                + "_IGM.py"
            )

        if return_all:
            return igm_hist
        else:
            if sim_igm not in igm_hist:
                raise ValueError(
                    sim_igm
                    + " string_split found in "
                    + fname
                    + r"\n Check out the LaCE script save_"
                    + sim_igm[:3]
                    + "_IGM.py"
                )
            else:
                return igm_hist[sim_igm]

    def set_priors(self, fid_igm, list_sim_cube, type_priors="hc"):
        """Set priors for all IGM models"""

        if type_priors == "hc":
            percent = 95
        elif type_priors == "data":
            percent = 68
        else:
            raise ValueError("type_priors must be 'hc' or 'data'")

        all_igm = self.get_igm(list_sim_cube[0], return_all=True)

        self.priors = {}
        for par in fid_igm:
            if par == "z":
                continue
            res_div = np.zeros((len(all_igm), 2))
            for ii, sim in enumerate(all_igm):
                string_split = sim.split("_")
                sim_label = string_split[0] + "_" + string_split[1]
                if sim_label not in list_sim_cube:
                    continue
                _ = np.argwhere((fid_igm[par] != 0) & (all_igm[sim][par] != 0))[
                    :, 0
                ]
                if len(_) == 0:
                    continue
                res_div[ii, 0] = np.abs(
                    np.max(all_igm[sim][par][_] / fid_igm[par][_])
                )
                res_div[ii, 1] = np.abs(
                    np.min(all_igm[sim][par][_] / fid_igm[par][_])
                )

            _ = np.argwhere(np.isfinite(res_div[:, 0]) & (res_div[:, 0] != 0))[
                :, 0
            ]
            y0_max = np.abs(np.log(np.percentile(res_div[_, 0], percent)))
            _ = np.argwhere(np.isfinite(res_div[:, 1]) & (res_div[:, 1] != 0))[
                :, 0
            ]
            y0_min = np.abs(np.log(np.percentile(1 / res_div[_, 1], percent)))
            y0_cen = 0.5 * (y0_max + y0_min)
            y1 = y0_cen / np.log((1 + fid_igm["z"].max()) / (1 + self.z_pivot))
            self.priors[par] = [[-y1, y1], [-y0_min * 1.05, y0_max * 1.05]]
            print(par, self.priors[par])

        # self.shift = {}
        # # adjust prior to fiducial IGM history
        # for par in cen_igm:
        #     if par == "z":
        #         continue

        #     res_div = np.abs(np.median(fid_igm[par] / cen_igm[par]))
        #     y0 = np.log(np.percentile(res_div, percent))
        #     self.priors[par] = y0
