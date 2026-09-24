import numpy as np
from scipy.interpolate import (
    make_smoothing_spline,
    make_interp_spline,
    interp1d,
)
from cup1d.likelihood import parameter as likelihood_parameter


class IGM_model(object):
    """New model for HCD contamination"""

    def __init__(
        self,
        coeffs=None,
        list_coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_igm=None,
        fid_vals=None,
        flat_priors=None,
        Gauss_priors=None,
    ):
        # store input data
        self.list_coeffs = list_coeffs
        self.z_0 = z_0
        self.fid_vals = fid_vals
        self.Gauss_priors = Gauss_priors
        self.flat_priors = flat_priors
        self.fid_interp = {}

        # set prop_coeffs (only for interp, not pivot)
        self.prop_coeffs = {}
        for key in self.list_coeffs:
            try:
                self.prop_coeffs[key + "_otype"] = prop_coeffs[key + "_otype"]
            except KeyError:
                raise ValueError("must specify otype in prop_coeffs for:", key)
            try:
                self.prop_coeffs[key + "_ztype"] = prop_coeffs[key + "_ztype"]
            except KeyError:
                raise ValueError("must specify ztype in prop_coeffs for:", key)

            if prop_coeffs[key + "_ztype"].startswith("interp"):
                try:
                    self.prop_coeffs[key + "_znodes"] = prop_coeffs[
                        key + "_znodes"
                    ]
                except KeyError:
                    raise ValueError(
                        "must specify znodes in prop_coeffs for:", key
                    )

        self.coeffs = {}
        if coeffs is not None:
            if free_param_names is not None:
                raise ValueError(
                    "can not specify both coeffs and free_param_names"
                )
            for key in self.list_coeffs:
                # set coeffs
                if key in coeffs:
                    self.coeffs[key] = coeffs[key]
                else:
                    raise ("Coeff not specified:", key)
        else:
            if free_param_names is None:
                raise ValueError(
                    "must specify either coeffs or free_param_names"
                )

            # figure out number of HCD free params
            self.n_pars = {}
            for key in self.list_coeffs:
                self.n_pars[key] = len(
                    [p for p in free_param_names if key + "_" in p]
                )
                if self.n_pars[key] == 0:
                    npar = 1
                else:
                    npar = self.n_pars[key]
                self.coeffs[key] = [0.0] * npar

                for ii in range(npar):
                    if self.prop_coeffs[key + "_ztype"] == "pivot":
                        if ii == 0:
                            self.coeffs[key][-1] = self.fid_vals[key][-1]
                        else:
                            self.coeffs[key][-(ii + 1)] = self.fid_vals[key][0]
                    else:
                        self.coeffs[key][ii] = self.fid_vals[key][ii]

        # post-process fiducial IGM
        for key in self.list_coeffs:
            self.process_igm(fid_igm, key)

        self.set_params()

    def process_igm(
        self,
        fid_igm,
        name_coeff,
        order_extra=2,
        smoothing=True,
        zmin=1.9,
        zmax=5.5,
    ):
        """Post-process IGM from simulation"""

        mask = (
            (fid_igm[name_coeff + "_z"] != 0)
            & (fid_igm[name_coeff] != 0)
            & np.isfinite(fid_igm[name_coeff])
        )
        mask_znonzero = fid_igm[name_coeff + "_z"] != 0
        if np.sum(mask) == 0:
            raise ValueError("No non-zero value for fiducial IGM", name_coeff)
        elif np.sum(mask) != fid_igm[name_coeff].shape[0]:
            print(
                "The fiducial value of",
                name_coeff,
                " is zero for z: ",
                fid_igm[name_coeff + "_z"][mask == False],
            )
        # print(name_coeff, fid_igm[name_coeff], fid_igm[name_coeff][mask])

        # fit to fiducial data to reduce noise
        y = fid_igm[name_coeff][mask]
        if self.prop_coeffs[name_coeff + "_otype"] == "exp":
            y = np.log(y)

        pfit = np.polyfit(fid_igm[name_coeff + "_z"][mask], y, order_extra)
        p = np.poly1d(pfit)

        # extrapolate to z=2 (if needed)
        if np.min(fid_igm[name_coeff + "_z"]) > zmin:
            z_to_inter = np.concatenate(
                [[zmin], fid_igm[name_coeff + "_z"][mask_znonzero]]
            )
        else:
            z_to_inter = fid_igm[name_coeff + "_z"][mask_znonzero]

        # extrapolate to z=5.0 (if needed)
        if np.max(fid_igm[name_coeff + "_z"]) < zmax:
            z_to_inter = np.concatenate([z_to_inter, [zmax]])
        else:
            z_to_inter = fid_igm[name_coeff + "_z"][mask_znonzero]

        if smoothing:
            fid_vals = p(z_to_inter)
            if self.prop_coeffs[name_coeff + "_otype"] == "exp":
                fid_vals = np.exp(fid_vals)
        else:
            vlow = p(zmin)
            if self.prop_coeffs[name_coeff + "_otype"] == "exp":
                vlow = np.exp(vlow)

            vhigh = p(zmax)
            if self.prop_coeffs[name_coeff + "_otype"] == "exp":
                vhigh = np.exp(vhigh)

            if np.min(fid_igm[name_coeff + "_z"]) > zmin:
                fid_vals = np.concatenate(
                    [vlow, fid_igm[name_coeff][mask_znonzero]]
                )
            else:
                fid_vals = fid_igm[name_coeff][mask_znonzero]
            if np.max(fid_igm[name_coeff + "_z"]) < zmax:
                fid_vals = np.concatenate([fid_vals, vhigh])

            mask_coeff0 = fid_vals == 0
            # use poly fit to interpolate when data is missing (needed for Nyx)
            fid_vals[mask_coeff0] = p(z_to_inter[mask_coeff0])
            if self.prop_coeffs[name_coeff + "_otype"] == "exp":
                fid_vals[mask_coeff0] = np.exp(fid_vals[mask_coeff0])

        # create interpolator
        ind = np.argsort(z_to_inter)
        self.fid_interp[name_coeff] = interp1d(
            z_to_inter[ind], fid_vals[ind], kind="cubic"
        )

    def set_params(self):
        """Setup likelihood parameters in the HCD model"""

        self.params = {}

        for key in self.list_coeffs:
            values = self.coeffs[key]
            for ii in range(len(values)):
                name = key + "_" + str(ii)
                set_prior = False
                for key2 in self.flat_priors:
                    if key2 in name:
                        if self.prop_coeffs[key + "_ztype"] == "pivot":
                            if ii == 0:
                                xmin = self.flat_priors[key2][-1][0]
                                xmax = self.flat_priors[key2][-1][1]
                            else:
                                xmin = self.flat_priors[key2][0][0]
                                xmax = self.flat_priors[key2][0][1]
                        else:
                            xmin = self.flat_priors[key2][-1][0]
                            xmax = self.flat_priors[key2][-1][1]
                        set_prior = True
                        break

                if set_prior is False:
                    raise ValueError("Cannot find priors of:", key)

                # note non-trivial order in coefficients
                Gwidth = None
                if self.Gauss_priors is not None:
                    if name in self.Gauss_priors:
                        if self.prop_coeffs[key + "_ztype"] == "pivot":
                            Gwidth = self.Gauss_priors[name][-(ii + 1)]
                        else:
                            Gwidth = self.Gauss_priors[name][ii]

                if self.prop_coeffs[key + "_ztype"] == "pivot":
                    _value = values[-(ii + 1)]
                else:
                    _value = values[ii]

                par = likelihood_parameter.make_parameter(
                    name=name,
                    value=_value,
                    min_value=xmin,
                    max_value=xmax,
                    Gauss_priors_width=Gwidth,
                    hessian_transform=self.prop_coeffs[key + "_otype"],
                )
                self.params[name] = par

    def get_Nparam(self):
        """Number of parameters in the model"""
        n_params = len(self.params)
        n_coeffs = 0
        for coeff in self.coeffs:
            n_coeffs += len(coeff)
        if n_params != n_coeffs:
            raise ValueError("mismatch between number of params and coeffs")
        return n_params

    def get_value(self, name, z, like_params=None):
        coeff = self.get_coeff(name, like_params=like_params)

        if self.prop_coeffs[name + "_ztype"] == "pivot":
            xz = np.log((1 + z) / (1 + self.z_0))
            ln_poly = np.poly1d(coeff)
            ln_out = ln_poly(xz)
        elif self.prop_coeffs[name + "_ztype"].startswith("interp"):
            if self.prop_coeffs[name + "_ztype"].endswith("_lin"):
                ln_out = np.interp(z, self.prop_coeffs[name + "_znodes"], coeff)
            elif self.prop_coeffs[name + "_ztype"].endswith("_spl"):
                f_out = make_interp_spline(
                    self.prop_coeffs[name + "_znodes"],
                    coeff,
                    k=1,
                )
                ln_out = f_out(z)
            elif self.prop_coeffs[name + "_ztype"].endswith("_smspl"):
                f_out = make_smoothing_spline(
                    self.prop_coeffs[name + "_znodes"], coeff
                )
                ln_out = f_out(z)
            else:
                raise ValueError(
                    "prop_coeffs must be interp_lin, interp_spl, or interp_smspl for",
                    name,
                )
        else:
            raise ValueError("prop_coeffs must be interp or pivot for", name)

        if self.prop_coeffs[name + "_otype"] == "const":
            return ln_out
        elif self.prop_coeffs[name + "_otype"] == "exp":
            return np.exp(ln_out)
        else:
            raise ValueError("prop_coeffs must be const or exp for", name)

    def get_parameter(self, name):
        return self.params[name]

    def get_parameters(self):
        """Return likelihood parameters"""
        return self.params

    def get_coeff(self, name, like_params=None):
        if like_params:
            coeff = self.coeffs[name].copy()
            Npar = 0
            array_names = []
            array_values = []
            for par_name, par_value in like_params.items():
                if (name + "_") in par_name:
                    array_names.append(par_name)
                    array_values.append(par_value)
                    Npar += 1
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # return fiducial value
            if Npar == 0:
                return coeff
            elif Npar != self.n_pars[name]:
                print(Npar, self.n_pars[name])
                raise ValueError("number of params mismatch for: " + name)

            for ii in range(Npar):
                ind_arr = np.argwhere(name + "_" + str(ii) == array_names)[0, 0]
                if self.prop_coeffs[name + "_ztype"] == "pivot":
                    coeff[-(ii + 1)] = array_values[ind_arr]
                else:
                    coeff[ii] = array_values[ind_arr]
        else:
            coeff = self.coeffs[name]

        return coeff

    def reset_coeffs(self, like_params, rank=0):
        """Reset all coefficients to fiducial values"""
        for name in self.coeffs:
            Npar = 0
            if rank == 0:
                print("orig", name, self.coeffs[name])
            array_names = []
            array_values = []
            for par_name, par_value in like_params.items():
                if (name + "_") in par_name:
                    array_names.append(par_name)
                    array_values.append(par_value)
                    Npar += 1
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # return fiducial value
            if Npar == 0:
                continue
            elif Npar != self.n_pars[name]:
                if rank == 0:
                    print(Npar, self.n_pars[name])
                raise ValueError("number of params mismatch for: " + name)

            for ii in range(Npar):
                ind_arr = np.argwhere(name + "_" + str(ii) == array_names)[0, 0]
                if self.prop_coeffs[name + "_ztype"] == "pivot":
                    self.coeffs[name][-(ii + 1)] = array_values[ind_arr]
                else:
                    self.coeffs[name][ii] = array_values[ind_arr]
            if rank == 0:
                print("new", name, self.coeffs[name])

    def plot_parameters(self, z, like_params, folder=None):
        """Delegate to :func:`cup1d.postprocessing.igm.plot_parameters`."""
        from cup1d.postprocessing.igm import plot_parameters as _plot

        return _plot(self, z, like_params, folder)
