class IGM(object):
    """Contains all IGM models"""

    def __init__(
        self,
        zs,
        free_param_names=None,
        F_model=None,
        T_model=None,
        P_model=None,
        fid_sim_igm="mpg_central",
    ):
        # load fiducial IGM history (used for fitting)
        self.fid_sim_igm = fid_sim_igm
        fid_igm = self.get_igm(fid_sim_igm)

        # setup fiducial IGM models
        if F_model is not None:
            self.F_model = F_model
        else:
            self.F_model = mean_flux_model.MeanFluxModel(
                free_param_names=free_param_names,
                fid_igm=fid_igm,
            )
        if T_model:
            self.T_model = T_model
        else:
            self.T_model = thermal_model.ThermalModel(
                free_param_names=free_param_names,
                fid_igm=fid_igm,
            )
        if P_model:
            self.P_model = P_model
        else:
            self.P_model = pressure_model.PressureModel(
                free_param_names=free_param_names,
                fid_igm=fid_igm,
            )

        self.fid_igm = {}
        self.fid_igm["z"] = zs
        self.fid_igm["tau_eff"] = self.F_model.get_tau_eff(zs)
        self.fid_igm["gamma"] = self.T_model.get_gamma(zs)
        self.fid_igm["sigT_kms"] = self.T_model.get_sigT_kms(zs)
        self.fid_igm["kF_kms"] = self.P_model.get_kF_kms(zs)

    def get_igm(self, sim_igm):
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
                + "not found. You can produce it using LaCE"
                + r"\n script save_"
                + sim_igm[:3]
                + "_IGM.py"
            )
        else:
            if sim_igm not in igm_hist:
                raise ValueError(
                    sim_igm
                    + " not found in "
                    + fname
                    + r"\n Check out the LaCE script save_"
                    + sim_igm[:3]
                    + "_IGM.py"
                )
            else:
                fid_igm = igm_hist[sim_igm]

        return fid_igm
