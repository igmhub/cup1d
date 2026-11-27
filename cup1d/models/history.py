"""Redshift histories of coefficient families.

Shared by the IGM models (:mod:`cup1d.models.igm`) and the contaminant
models (:mod:`cup1d.models.contaminants`): each family (``tau_eff``,
``f_Lya_SiIII``, ...) is a polynomial in ``ln((1+z)/(1+z_0))`` (``pivot``) or
an interpolation between redshift nodes (``interp_lin``, ``interp_spl``,
``interp_smspl``), optionally exponentiated (``otype: exp``). Histories are
evaluated for a batch of parameter points; one point is a batch of one.
"""

import numpy as np
from scipy.interpolate import make_interp_spline, make_smoothing_spline

from cup1d.likelihood import parameter as likelihood_parameter
from cup1d.utils.batch import as_columns
from cup1d.utils.utils import split_string


class CoefficientHistories:
    """Parameters and redshift histories of coefficient families.

    Subclasses set ``list_coeffs``, ``coeffs``, ``n_pars``, ``prop_coeffs``,
    ``z_0``, ``flat_priors`` and ``Gauss_priors`` (and optionally ``z_max``
    with ``null_vals``).
    """

    def set_coefficients(self, free_param_names):
        """Number of free coefficients and fiducial coefficients of every family.

        Parameters
        ----------
        free_param_names : sequence of str
            Free parameters; ``<family>_<index>`` sets ``n_pars[family]``.

        Notes
        -----
        A pivot family gets its constant term from the last fiducial value
        and its higher orders from the first. An interpolated family gets one
        coefficient per free node: the fiducial value of that node when the
        fiducial has one value per node, else its (last) single value. A
        family without free coefficients keeps one fixed coefficient.
        """
        self.n_pars = {}
        for key in self.list_coeffs:
            self.n_pars[key] = len([p for p in free_param_names if split_string(p)[0] == key])
            npar = max(self.n_pars[key], 1)
            fiducial = np.atleast_1d(np.asarray(self.fid_vals[key], dtype=float))
            coeffs = [0.0] * npar
            for ii in range(npar):
                if self.prop_coeffs[key + "_ztype"] == "pivot":
                    coeffs[-(ii + 1)] = fiducial[-1] if ii == 0 else fiducial[0]
                else:
                    coeffs[ii] = fiducial[ii] if len(fiducial) == npar else fiducial[-1]
            self.coeffs[key] = coeffs

    def get_value(self, name, z, like_params=None):
        """History of one family at ``z`` for one point or a batch.

        Parameters
        ----------
        name : str
            Coefficient family.
        z : float or array_like
            Redshift(s).
        like_params : mapping, optional
            Scalar values (one point) or columns of shape ``(n_batch,)``;
            families without values keep their stored coefficients.

        Returns
        -------
        float or ndarray
            For one point, the shape of ``z``; for a batch,
            ``(n_batch, n_z)``. See :meth:`get_value_batch`.
        """
        columns, n_batch, scalar = as_columns(like_params)
        out = self.get_value_batch(name, z, columns, n_batch=n_batch)
        if not scalar:
            return out
        out = out[0]
        return out[0] if np.ndim(z) == 0 else out

    def get_value_batch(self, name, z, like_params, n_batch=None):
        """History of one family for a batch of parameter points.

        Parameters
        ----------
        name : str
            Coefficient family.
        z : float or array_like
            Redshift(s).
        like_params : mapping of str to ndarray
            Columns of shape ``(n_batch,)``. Coefficients of the family that
            are absent keep their stored values.
        n_batch : int, optional
            Batch size; from the columns if omitted (1 for an empty mapping).

        Returns
        -------
        ndarray, shape (n_batch, n_z)
            The history after the output transform (``exp`` or ``const``).
            Above ``z_max[name]`` (if set) the history is the null value.
            This is the multiplier of the fiducial history for IGM families.

        Raises
        ------
        ValueError
            For a column of the wrong shape or an unknown history type.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        if n_batch is None:
            n_batch = len(next(iter(like_params.values()))) if like_params else 1
        ztype = self.prop_coeffs[name + "_ztype"]
        pivot = ztype == "pivot"
        stored = np.asarray(self.coeffs[name], dtype=float)
        coeff = np.broadcast_to(stored, (n_batch, len(stored))).copy()
        for index in range(self.n_pars[name]):
            parameter_name = name + "_" + str(index)
            if parameter_name not in like_params:
                continue
            value = np.asarray(like_params[parameter_name], dtype=float)
            if value.shape != (n_batch,):
                raise ValueError(
                    f"batched parameter {parameter_name} has shape {value.shape}; "
                    f"expected ({n_batch},)"
                )
            # pivot coefficient 0 is the constant term, stored last
            coeff[:, -(index + 1) if pivot else index] = value

        if pivot:
            xz = np.log((1 + z) / (1 + self.z_0))
            ln_out = np.zeros((n_batch, len(z)))
            for coefficient in coeff.T:  # Horner, as numpy.polyval
                ln_out = ln_out * xz[None, :] + coefficient[:, None]
        elif ztype.startswith("interp"):
            nodes = self.prop_coeffs[name + "_znodes"]
            if ztype.endswith("_lin"):
                ln_out = np.asarray([np.interp(z, nodes, row) for row in coeff], dtype=float)
            elif ztype.endswith("_spl"):
                spline = make_interp_spline(nodes, coeff.T, k=1, axis=0)
                ln_out = np.asarray(spline(z)).T
            elif ztype.endswith("_smspl"):
                ln_out = np.asarray(
                    [make_smoothing_spline(nodes, row)(z) for row in coeff], dtype=float
                )
            else:
                raise ValueError(
                    f"ztype must be interp_lin, interp_spl or interp_smspl for {name}"
                )
        else:
            raise ValueError(f"ztype must be pivot or interp_* for {name}")
        ln_out = np.array(ln_out, dtype=float).reshape(n_batch, len(z))

        z_max = getattr(self, "z_max", None)
        if z_max is not None and name in z_max:
            ln_out[:, z >= z_max[name]] = self.null_vals[name]

        otype = self.prop_coeffs[name + "_otype"]
        if otype == "const":
            return ln_out
        if otype == "exp":
            return np.exp(ln_out)
        raise ValueError(f"otype must be const or exp for {name}")

    def set_params(self):
        """Build canonical likelihood parameter definitions for coefficients.

        Returns
        -------
        None
            Sets ``params`` keyed by indexed coefficient parameter name.

        Raises
        ------
        ValueError
            If no matching flat-prior definition exists for a coefficient.
        """

        self.params = {}

        for key in self.list_coeffs:
            values = self.coeffs[key]
            for ii in range(len(values)):
                name = key + "_" + str(ii)
                set_prior = False
                for key2 in self.flat_priors:
                    if key2 == key:
                        if self.prop_coeffs[key + "_ztype"] == "pivot":
                            # xmin = self.flat_priors[key2][-(ii + 1)][0]
                            # xmax = self.flat_priors[key2][-(ii + 1)][1]
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
        """Return the number of coefficient likelihood parameters.

        Returns
        -------
        int
            Number of entries in ``params``.

        Raises
        ------
        ValueError
            If the parameter count and stored coefficient count disagree.
        """
        n_params = len(self.params)
        n_coeffs = 0
        for coeff in self.coeffs:
            n_coeffs += len(coeff)
        if n_params != n_coeffs:
            raise ValueError("mismatch between number of params and coeffs")
        return n_params

    def get_parameter(self, name):
        """Return a stored coefficient parameter definition.

        Parameters
        ----------
        name : str
            Full coefficient parameter name in ``self.params``.

        Returns
        -------
        dict
            Shared parameter definition. Mutations affect the model.

        Raises
        ------
        KeyError
            If the parameter name is absent.
        """
        return self.params[name]

    def get_parameters(self):
        """Return shared likelihood parameter definitions for all families.

        Returns
        -------
        dict
            Mapping from indexed coefficient names to mutable parameter records.
        """
        return self.params

    def get_coeff(self, name, like_params=None):
        """Retrieve history coefficients with physical named overrides.

        Parameters
        ----------
        name : str
            Coefficient family in ``self.coeffs``.
        like_params : mapping, optional
            Values keyed by ``name_<index>``. If the family is overridden,
            all configured indices must be supplied. Other families are ignored.

        Returns
        -------
        numpy.ndarray
            Coefficients in polynomial order for pivot histories or node order
            for interpolation. Overrides operate on a copy; an absent or empty
            mapping returns the stored array itself.

        Raises
        ------
        ValueError
            If the number of supplied family coefficients is inconsistent.

        Notes
        -----
        Pivot coefficient index zero denotes the constant polynomial term,
        stored at the end of the array.
        """
        if like_params:
            coeff = self.coeffs[name].copy()
            Npar = 0
            array_names = []
            array_values = []
            for par_name, par_value in like_params.items():
                if split_string(par_name)[0] == name:
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
        """Persist complete named coefficient overrides into this model.

        Parameters
        ----------
        like_params : mapping
            Physical coefficient values keyed by ``<family>_<index>``. A
            partially supplied family is rejected to preserve its history.
        rank : int, default=0
            MPI rank that emits before/after diagnostic output.

        Returns
        -------
        None
            Updates ``coeffs`` in place.

        Raises
        ------
        ValueError
            If supplied values cover only part of a coefficient family.
        """
        for name in self.coeffs:
            Npar = 0
            if rank == 0:
                print("orig", name, self.coeffs[name])
            array_names = []
            array_values = []
            for par_name, par_value in like_params.items():
                if split_string(par_name)[0] == name:
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
