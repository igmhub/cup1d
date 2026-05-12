"""Intergalactic Medium (IGM) modeling module.

This module provides classes for modeling the IGM properties including
temperature, pressure, and mean flux evolution.

"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import numpy.typing as npt
from scipy.interpolate import (
    interp1d,
    make_interp_spline,
    make_smoothing_spline,
)

from cup1d.likelihood import likelihood_parameter

# Type aliases
Array1D = npt.NDArray[np.float64]
Array2D = npt.NDArray[np.float64]
Float = float | int


class IGM_model:
    """Base model for redshift-dependent IGM nuisance parameters.

    Parameters
    ----------
    coeffs : Optional[Dict[str, float]], optional
        Coefficient dictionary.
    list_coeffs : Optional[List[str]], optional
        List of coefficient names.
    prop_coeffs : Optional[Dict[str, Any]], optional
        Coefficient properties.
    free_param_names : Optional[List[str]], optional
        List of free parameter names.
    z_0 : float, optional
        Pivot redshift.
    fid_igm : Optional[Dict[str, Array1D]], optional
        Fiducial IGM parameters.
    fid_vals : Optional[Dict[str, Array1D]], optional
        Fiducial values.
    flat_priors : Optional[Dict[str, Tuple[float, float]]], optional
        Flat prior bounds.
    Gauss_priors : Optional[Dict[str, float]], optional
        Gaussian prior widths.
    """

    def __init__(
        self,
        coeffs: dict[str, float] | None = None,
        list_coeffs: list[str] | None = None,
        prop_coeffs: dict[str, Any] | None = None,
        free_param_names: list[str] | None = None,
        z_0: float = 3.0,
        fid_igm: dict[str, Array1D] | None = None,
        fid_vals: dict[str, Array1D] | None = None,
        flat_priors: dict[str, tuple[float, float]] | None = None,
        Gauss_priors: dict[str, float] | None = None,
    ) -> None:
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
                raise ValueError("must specify otype in prop_coeffs for:", key) from None
            try:
                self.prop_coeffs[key + "_ztype"] = prop_coeffs[key + "_ztype"]
            except KeyError:
                raise ValueError("must specify ztype in prop_coeffs for:", key) from None

            if prop_coeffs[key + "_ztype"].startswith("interp"):
                try:
                    self.prop_coeffs[key + "_znodes"] = prop_coeffs[key + "_znodes"]
                except KeyError:
                    raise ValueError("must specify znodes in prop_coeffs for:", key) from None

        self.coeffs = {}
        if coeffs is not None:
            if free_param_names is not None:
                raise ValueError("can not specify both coeffs and free_param_names")
            for key in self.list_coeffs:
                # set coeffs
                if key in coeffs:
                    self.coeffs[key] = coeffs[key]
                else:
                    raise ValueError(f"Coeff not specified: {key}")
        else:
            if free_param_names is None:
                raise ValueError("must specify either coeffs or free_param_names")

            # figure out number of IGM free params
            self.n_pars = {}
            for key in self.list_coeffs:
                self.n_pars[key] = len([p for p in free_param_names if key + "_" in p])
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
        fid_igm: dict[str, Array1D],
        name_coeff: str,
        order_extra: int = 2,
        smoothing: bool = True,
        zmin: float = 1.9,
        zmax: float = 5.5,
    ) -> None:
        """Post-process IGM from simulation.

        Parameters
        ----------
        fid_igm : Dict[str, Array1D]
            Fiducial IGM parameters dictionary.
        name_coeff : str
            Name of the coefficient to process.
        order_extra : int, optional
            Polynomial order for fitting.
        smoothing : bool, optional
            Whether to apply smoothing.
        zmin : float, optional
            Minimum redshift for extrapolation.
        zmax : float, optional
            Maximum redshift for extrapolation.
        """

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
                fid_igm[name_coeff + "_z"][not mask],
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
                fid_vals = np.concatenate([vlow, fid_igm[name_coeff][mask_znonzero]])
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

    def set_params(self) -> None:
        """Create likelihood parameters for all IGM coefficients."""
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

                par = likelihood_parameter.LikelihoodParameter(
                    name=name,
                    value=_value,
                    min_value=xmin,
                    max_value=xmax,
                    Gauss_priors_width=Gwidth,
                )
                self.params[name] = par

    def get_Nparam(self) -> int:
        """Number of parameters in the model.

        Returns
        -------
        int
            Number of parameters.
        """
        n_params = len(self.params)
        n_coeffs = 0
        for coeff in self.coeffs:
            n_coeffs += len(self.coeffs[coeff])
        if n_params != n_coeffs:
            raise ValueError("mismatch between number of params and coeffs")
        return n_params

    def get_value(self, name: str, z: float, like_params: list = None) -> float:
        """Evaluate one IGM coefficient at redshift ``z``.

        The returned value is either the evolved coefficient itself or its
        exponential, depending on ``prop_coeffs[f"{name}_otype"]``.
        """
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
                f_out = make_smoothing_spline(self.prop_coeffs[name + "_znodes"], coeff)
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
        """Return one likelihood parameter by name."""
        return self.params[name]

    def get_parameters(self):
        """Return all likelihood parameters."""
        return self.params

    def get_coeff(self, name, like_params=None):
        """Return coefficients for ``name``, optionally updated from parameters."""
        if like_params:
            coeff = self.coeffs[name].copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if (name + "_") in par.name:
                    array_names.append(par.name)
                    array_values.append(par.value)
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
        """Update stored coefficients from a list of likelihood parameters."""
        for name in self.coeffs:
            Npar = 0
            if rank == 0:
                print("orig", name, self.coeffs[name])
            array_names = []
            array_values = []
            for par in like_params:
                if (name + "_") in par.name:
                    array_names.append(par.name)
                    array_values.append(par.value)
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
        """Plot IGM parameter evolution over redshift."""

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(
            len(self.coeffs), 1, sharex=True, figsize=(8, 3 * len(self.coeffs))
        )
        if len(self.coeffs) == 1:
            ax = [ax]

        try:
            len(like_params[0])
        except Exception:
            z_at_time = False
        else:
            z_at_time = True

        vals_out = {}
        coeffs_out = {}

        for ii, key in enumerate(self.coeffs.keys()):
            if z_at_time is False:
                if key == "tau_eff":
                    vals = self.get_tau_eff(z, like_params=like_params)
                elif key == "gamma":
                    vals = self.get_gamma(z, like_params=like_params)
                elif key == "sigT_kms":
                    vals = self.get_sigT_kms(z, like_params=like_params)
                elif key == "kF_kms":
                    vals = self.get_kF_kms(z, like_params=like_params)
                else:
                    raise ValueError("key must be tau_eff, gamma, sigT_kms, or kF_kms")
                coeffs_out[key] = self.get_coeff(key, like_params=like_params)
            else:
                vals = []
                coeffs_out[key] = []
                for jj in range(len(z)):
                    if key == "tau_eff":
                        vals.append(
                            self.get_tau_eff(z[jj], like_params=like_params[jj])
                        )
                    elif key == "gamma":
                        vals.append(self.get_gamma(z[jj], like_params=like_params[jj]))
                    elif key == "sigT_kms":
                        vals.append(
                            self.get_sigT_kms(z[jj], like_params=like_params[jj])
                        )
                    elif key == "kF_kms":
                        vals.append(self.get_kF_kms(z[jj], like_params=like_params[jj]))
                    else:
                        raise ValueError(
                            "key must be tau_eff, gamma, sigT_kms, or kF_kms"
                        )
                    coeffs_out[key].append(
                        self.get_coeff(key, like_params=like_params[jj])[0]
                    )
                vals = np.array(vals)

            if key == "tau_eff":
                fid_vals = self.get_tau_eff(z)
            elif key == "gamma":
                fid_vals = self.get_gamma(z)
            elif key == "sigT_kms":
                fid_vals = self.get_sigT_kms(z)
            elif key == "kF_kms":
                fid_vals = self.get_kF_kms(z)

            if self.prop_coeffs[key + "_otype"] == "exp":
                vals = np.log(vals)
                fid_vals = np.log(fid_vals)

            vals_out[key] = vals

            ax[ii].plot(z, vals, "o-", label="data")
            res = np.polyfit(z, vals, 1)
            ax[ii].plot(z, res[0] * z + res[1], "--", label="fit")
            ax[ii].plot(z, fid_vals, "-.", label="fid")
            ax[ii].set_ylabel(key)
        ax[0].legend()
        ax[-1].set_xlabel("z")

        plt.tight_layout()
        plt.show()

        if folder is not None:
            fig.savefig(folder + ".png")
            fig.savefig(folder + ".pdf")

        return vals_out, coeffs_out
