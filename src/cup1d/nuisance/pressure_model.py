import numpy as np
import copy
import os
from scipy.interpolate import interp1d
from cup1d.likelihood import likelihood_parameter

# lambda_F ~ 80 kpc ~ 0.08 Mpc ~ 0.055 Mpc/h ~ 5.5 km/s (Onorbe et al. 2016)
# k_F = 1 / lambda_F ~ 12.5 1/Mpc ~ 18.2 h/Mpc ~ 0.182 s/km


class PressureModel(object):
    """Model the redshift evolution of the pressure smoothing length.
    We use a power law rescaling around a fiducial simulation at the centre
    of the initial Latin hypercube in simulation space."""

    def __init__(
        self,
        z_kF=3.0,
        ln_kF_coeff=None,
        free_param_names=None,
        fid_igm=None,
    ):
        """Construct model with central redshift and (x2,x1,x0) polynomial."""

        if fid_igm is None:
            fname = (
                os.environ["LACE_REPO"]
                + "/src/lace/data/sim_suites/Australia20/IGM_histories.npy"
            )
            try:
                igm_hist = np.load(fname, allow_pickle=True).item()
            except:
                raise ValueError(
                    fname
                    + " not found. You can produce it using the LaCE"
                    + r" script save_mpg_IGM.py"
                )
            else:
                fid_igm = igm_hist["mpg_central"]
        self.fid_igm = fid_igm

        mask = fid_igm["tau_eff"] != 0
        if np.sum(mask) != fid_igm["tau_eff"].shape[0]:
            print(
                "The fiducial value of tau_eff is zero for z: ",
                fid_igm["z"][mask == False],
            )
        if np.sum(mask) > 0:
            self.fid_z = fid_igm["z"][mask]
            self.fid_tau_eff = fid_igm["tau_eff"][mask]
        else:
            raise ValueError("No non-zero tau_eff in fiducial IGM")

        mask = fid_igm["kF_kms"] != 0
        if np.sum(mask) != fid_igm["kF_kms"].shape[0]:
            print(
                "The fiducial value of kF is zero for z: ",
                fid_igm["z"][mask == False],
            )
        if np.sum(mask) > 0:
            self.fid_z = fid_igm["z"][mask]
            self.fid_kF = fid_igm["kF_kms"][mask]
        else:
            raise ValueError("No non-zero kF in fiducial IGM")

        self.fid_kF_interp = interp1d(self.fid_z, self.fid_kF, kind="cubic")

        self.z_kF = z_kF
        if ln_kF_coeff:
            assert free_param_names is None
            self.ln_kF_coeff = ln_kF_coeff
        else:
            if free_param_names:
                # figure out number of free pressure params
                n_kF = len([p for p in free_param_names if "ln_kF_" in p])
            else:
                n_kF = 2
            self.ln_kF_coeff = [0.0] * n_kF
        # store list of likelihood parameters (might be fixed or free)
        self.set_parameters()

    def get_Nparam(self):
        """Number of parameters in the model"""
        assert len(self.ln_kF_coeff) == len(self.params), "size mismatch"
        return len(self.ln_kF_coeff)

    def power_law_scaling(self, z):
        """Power law rescaling around z_tau"""
        xz = np.log((1 + z) / (1 + self.z_kF))
        ln_poly = np.poly1d(self.ln_kF_coeff)
        ln_out = ln_poly(xz)
        return np.exp(ln_out)

    def get_kF_kms(self, z):
        """kF_kms at the input redshift"""
        kF_kms = self.power_law_scaling(z) * self.fid_kF_interp(z)
        return kF_kms

    def set_parameters(self):
        """Setup likelihood parameters in the pressure model"""

        self.params = []
        Npar = len(self.ln_kF_coeff)
        for i in range(Npar):
            name = "ln_kF_" + str(i)
            if i == 0:
                xmin = -0.8
                xmax = 0.8
            elif i == 1:
                xmin = -2.0
                xmax = 2.0
            else:
                xmin = -2.0
                xmax = 2.0
            # note non-trivial order in coefficients
            value = self.ln_kF_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.params.append(par)

        return

    def get_parameters(self):
        """Return likelihood parameters for the pressure model"""

        return self.params

    def update_parameters(self, like_params):
        """Look for pressure parameters in list of parameters"""

        Npar = self.get_Nparam()

        # loop over likelihood parameters
        for like_par in like_params:
            if "ln_kF" not in like_par.name:
                continue
            # make sure you find the parameter
            found = False
            # loop over parameters in pressure model
            for ip in range(len(self.params)):
                if self.params[ip].name == like_par.name:
                    assert found == False, "can not update parameter twice"
                    self.ln_kF_coeff[Npar - ip - 1] = like_par.value
                    found = True
            assert found == True, "could not update parameter " + like_par.name

        return

    def get_new_model(self, like_params=[]):
        """Return copy of model, updating values from list of parameters"""

        kF = PressureModel(
            fid_igm=self.fid_igm,
            z_kF=self.z_kF,
            ln_kF_coeff=copy.deepcopy(self.ln_kF_coeff),
        )
        kF.update_parameters(like_params)
        return kF
