"""Contaminant modeling for the Lyman-alpha forest.

This module provides classes for modeling metal-line contaminants
and HCD systems in the Lyman-alpha forest.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.interpolate import (
    make_interp_spline,
    make_smoothing_spline,
)

from cup1d.likelihood import likelihood_parameter

# Type aliases
Array1D = npt.NDArray[np.float64]


class Contaminant:
    """Base class for contaminant modeling.

    This class handles the evolution and interpolation of nuisance parameters
    associated with contaminants (e.g., HCDs, metal lines).

    Parameters
    ----------
    coeffs : dict[str, float] | None, optional
        Coefficient dictionary. Default is None.
    list_coeffs : list[str] | None, optional
        List of coefficient names. Default is None.
    prop_coeffs : dict[str, Any] | None, optional
        Coefficient properties. Default is None.
    free_param_names : list[str] | None, optional
        List of free parameter names. Default is None.
    z_0 : float, optional
        Pivot redshift. Default is 3.0.
    fid_vals : dict[str, Array1D] | None, optional
        Fiducial values. Default is None.
    null_vals : dict[str, float] | None, optional
        Null values for baseline. Default is None.
    z_max : dict[str, float] | None, optional
        Maximum redshift for each coefficient. Default is None.
    flat_priors : dict[str, list[list[float]]] | None, optional
        Flat prior bounds. Default is None.
    Gauss_priors : dict[str, list[float]] | None, optional
        Gaussian prior widths. Default is None.

    Attributes
    ----------
    list_coeffs : list[str] | None
        List of coefficient names.
    z_0 : float
        Pivot redshift.
    fid_vals : dict[str, Array1D] | None
        Fiducial values.
    null_vals : dict[str, float] | None
        Null values for baseline.
    Gauss_priors : dict[str, list[float]] | None
        Gaussian prior widths.
    flat_priors : dict[str, list[list[float]]] | None
        Flat prior bounds.
    z_max : dict[str, float] | None
        Maximum redshift for each coefficient.
    prop_coeffs : dict[str, Any]
        Coefficient properties.
    coeffs : dict[str, list[float]]
        Coefficient values.
    n_pars : dict[str, int]
        Number of parameters for each coefficient.
    params : dict[str, likelihood_parameter.LikelihoodParameter]
        Likelihood parameters.
    """

    def __init__(
        self,
        coeffs: dict[str, float] | None = None,
        list_coeffs: list[str] | None = None,
        prop_coeffs: dict[str, Any] | None = None,
        free_param_names: list[str] | None = None,
        z_0: float = 3.0,
        fid_vals: dict[str, Array1D] | None = None,
        null_vals: dict[str, float] | None = None,
        z_max: dict[str, float] | None = None,
        flat_priors: dict[str, list[list[float]]] | None = None,
        Gauss_priors: dict[str, list[float]] | None = None,
    ) -> None:
        """Initialize the Contaminant model."""
        # store input data
        self.list_coeffs = list_coeffs
        self.z_0 = z_0
        self.fid_vals = fid_vals
        self.null_vals = null_vals
        self.Gauss_priors = Gauss_priors
        self.flat_priors = flat_priors
        self.z_max = z_max

        if self.list_coeffs is None:
            self.list_coeffs = []

        # set prop_coeffs (only for interp, not pivot)
        self.prop_coeffs = {}
        if prop_coeffs is not None:
            for key in self.list_coeffs:
                try:
                    self.prop_coeffs[key + "_otype"] = prop_coeffs[key + "_otype"]
                except KeyError:
                    raise ValueError(
                        "must specify otype in prop_coeffs for:", key
                    ) from None
                try:
                    self.prop_coeffs[key + "_ztype"] = prop_coeffs[key + "_ztype"]
                except KeyError:
                    raise ValueError(
                        "must specify ztype in prop_coeffs for:", key
                    ) from None

                if prop_coeffs[key + "_ztype"].startswith("interp"):
                    try:
                        self.prop_coeffs[key + "_znodes"] = prop_coeffs[key + "_znodes"]
                    except KeyError:
                        raise ValueError(
                            "must specify znodes in prop_coeffs for:", key
                        ) from None

        self.coeffs = {}
        self.n_pars = {}
        if coeffs is not None:
            if free_param_names is not None:
                raise ValueError("can not specify both coeffs and free_param_names")
            for key in self.list_coeffs:
                # set coeffs
                if key in coeffs:
                    # Expecting coeffs to be a dict of lists or single values
                    val = coeffs[key]
                    if isinstance(val, (list, np.ndarray)):
                        self.coeffs[key] = list(val)
                    else:
                        self.coeffs[key] = [float(val)]
                    self.n_pars[key] = len(self.coeffs[key])
                else:
                    raise ValueError(f"Coeff not specified: {key}")
        else:
            if free_param_names is None:
                raise ValueError("must specify either coeffs or free_param_names")

            # figure out number of free params
            for key in self.list_coeffs:
                self.n_pars[key] = len([p for p in free_param_names if key + "_" in p])
                if self.n_pars[key] == 0:
                    npar = 1
                else:
                    npar = self.n_pars[key]
                self.coeffs[key] = [0.0] * npar

                if self.fid_vals is not None and key in self.fid_vals:
                    for ii in range(npar):
                        if self.prop_coeffs[key + "_ztype"] == "pivot":
                            if ii == 0:
                                self.coeffs[key][-1] = self.fid_vals[key][-1]
                            else:
                                self.coeffs[key][-(ii + 1)] = self.fid_vals[key][0]
                        else:
                            self.coeffs[key][ii] = self.fid_vals[key][-1]

        self.set_params()

    def set_params(self) -> None:
        """Create likelihood parameters for all contaminant coefficients."""
        self.params = {}

        if self.flat_priors is None:
            return

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
        """Return the number of likelihood parameters in the model.

        Returns
        -------
        int
            Number of parameters.

        Raises
        ------
        ValueError
            If there is a mismatch between number of parameters and coefficients.
        """
        n_params = len(self.params)
        n_coeffs = 0
        for coeff in self.coeffs:
            n_coeffs += len(self.coeffs[coeff])
        if n_params != n_coeffs:
            raise ValueError("mismatch between number of params and coeffs")
        return n_params

    def get_value(
        self,
        name: str,
        z: float | Array1D,
        like_params: list | None = None,
    ) -> float | Array1D:
        """Evaluate one nuisance coefficient at redshift ``z``.

        The interpolation/evolution mode is controlled by
        ``prop_coeffs[f"{name}_ztype"]``. Coefficients can be returned directly
        or exponentiated according to ``prop_coeffs[f"{name}_otype"]``.

        Parameters
        ----------
        name : str
            Coefficient name.
        z : float | Array1D
            Redshift(s).
        like_params : list | None, optional
            Likelihood parameters. Default is None.

        Returns
        -------
        float | Array1D
            Evaluated coefficient value(s).

        Raises
        ------
        ValueError
            If prop_coeffs are invalid.
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
            raise ValueError(
                "prop_coeffs must be interp_lin, interp_spl, interp_smspl, or pivot for",
                name,
            )

        if self.z_max is not None and name in self.z_max:
            if isinstance(z, np.ndarray) and len(z) > 1:
                _ = z >= self.z_max[name]
                if np.any(_):
                    if self.null_vals is not None and name in self.null_vals:
                        ln_out[_] = self.null_vals[name]
                    else:
                        ln_out[_] = 0.0
            else:
                if z >= self.z_max[name]:
                    if self.null_vals is not None and name in self.null_vals:
                        ln_out = self.null_vals[name]
                    else:
                        ln_out = 0.0

        if self.prop_coeffs[name + "_otype"] == "const":
            return ln_out
        elif self.prop_coeffs[name + "_otype"] == "exp":
            return np.exp(ln_out)
        else:
            raise ValueError("prop_coeffs must be const or exp for", name)

    def get_parameter(self, name: str) -> likelihood_parameter.LikelihoodParameter:
        """Return one likelihood parameter by name.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        likelihood_parameter.LikelihoodParameter
            The requested parameter.
        """
        return self.params[name]

    def get_parameters(self) -> dict[str, likelihood_parameter.LikelihoodParameter]:
        """Return all likelihood parameters.

        Returns
        -------
        dict[str, likelihood_parameter.LikelihoodParameter]
            Dictionary of likelihood parameters.
        """
        return self.params

    def get_coeff(self, name: str, like_params: list | None = None) -> list[float]:
        """Return coefficients for ``name``, optionally updated from a chain state.

        Parameters
        ----------
        name : str
            Coefficient name.
        like_params : list | None, optional
            Likelihood parameters. Default is None.

        Returns
        -------
        list[float]
            Coefficient values.

        Raises
        ------
        ValueError
            If number of parameters mismatch.
        """
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
            array_names_np = np.array(array_names)
            array_values_np = np.array(array_values)

            # return fiducial value
            if Npar == 0:
                return coeff
            elif Npar != self.n_pars[name]:
                print(Npar, self.n_pars[name])
                raise ValueError("number of params mismatch for: " + name)

            for ii in range(Npar):
                match = np.argwhere(name + "_" + str(ii) == array_names_np)
                if match.size == 0:
                    continue
                ind_arr = match[0, 0]
                if self.prop_coeffs[name + "_ztype"] == "pivot":
                    coeff[-(ii + 1)] = array_values_np[ind_arr]
                else:
                    coeff[ii] = array_values_np[ind_arr]
        else:
            coeff = self.coeffs[name]

        return coeff

    def reset_coeffs(self, like_params: list, rank: int = 0) -> None:
        """Update stored coefficients from a list of likelihood parameters.

        Parameters
        ----------
        like_params : list
            Likelihood parameters.
        rank : int, optional
            MPI rank. Default is 0.
        """
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
            array_names_np = np.array(array_names)
            array_values_np = np.array(array_values)

            # return fiducial value
            if Npar == 0:
                continue
            elif Npar != self.n_pars[name]:
                if rank == 0:
                    print(Npar, self.n_pars[name])
                raise ValueError("number of params mismatch for: " + name)

            for ii in range(Npar):
                match = np.argwhere(name + "_" + str(ii) == array_names_np)
                if match.size == 0:
                    continue
                ind_arr = match[0, 0]
                if self.prop_coeffs[name + "_ztype"] == "pivot":
                    self.coeffs[name][-(ii + 1)] = array_values_np[ind_arr]
                else:
                    self.coeffs[name][ii] = array_values_np[ind_arr]
            if rank == 0:
                print("new", name, self.coeffs[name])

    def plot_parameters(
        self,
        z: Array1D,
        like_params: list,
        folder: str | None = None,
    ) -> tuple[dict[str, Array1D], dict[str, Any]]:
        """Plot coefficient evolution over redshift.

        Parameters
        ----------
        z : Array1D
            Redshifts.
        like_params : list
            Likelihood parameters.
        folder : str | None, optional
            Folder to save plots. Default is None.

        Returns
        -------
        tuple[dict[str, Array1D], dict[str, Any]]
            Evaluated values and coefficients.
        """
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(
            len(self.coeffs), 1, sharex=True, figsize=(8, 3 * len(self.coeffs))
        )
        if len(self.coeffs) == 1:
            ax = [ax]

        try:
            len(like_params[0])
            z_at_time = True
        except (TypeError, IndexError):
            z_at_time = False

        vals_out = {}
        coeffs_out = {}

        for ii, key in enumerate(self.coeffs.keys()):
            if z_at_time is False:
                vals = self.get_value(key, z, like_params=like_params)
                coeffs_out[key] = self.get_coeff(key, like_params=like_params)
            else:
                vals_list = []
                coeffs_out[key] = []
                for jj in range(len(z)):
                    vals_list.append(
                        self.get_value(key, z[jj], like_params=like_params[jj])
                    )
                    coeffs_out[key].append(
                        self.get_coeff(key, like_params=like_params[jj])[0]
                    )
                vals = np.array(vals_list)

            if self.null_vals is not None and key in self.null_vals:
                if np.all(vals == self.null_vals[key]):
                    continue
            elif key == "HCD_const":
                if np.all(vals == 0):
                    continue

            if self.prop_coeffs[key + "_otype"] == "exp":
                vals = np.log(vals)

            vals_out[key] = vals

            mask = np.ones_like(vals, dtype=bool)
            if self.null_vals is not None and key in self.null_vals:
                mask = vals != self.null_vals[key]

            if np.any(mask):
                ax[ii].plot(z[mask], vals[mask], "o-", label="data")
                xz = np.log((1 + z) / (1 + self.z_0))
                res = np.polyfit(xz[mask], vals[mask], 1)
                ax[ii].plot(z[mask], res[0] * xz[mask] + res[1], "--", label="fit")
                ax[ii].set_ylabel(key)

        ax[0].legend()
        ax[-1].set_xlabel("z")

        plt.tight_layout()
        plt.show()
        if folder is not None:
            fig.savefig(folder + ".png")
            fig.savefig(folder + ".pdf")

        return vals_out, coeffs_out
