import numpy as np
from scipy.interpolate import interp1d
from cup1d.models.history import CoefficientHistories


class IGM_model(CoefficientHistories):
    """Base redshift-history model for one physical IGM quantity."""

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
        """Initialize coefficient histories and a smooth fiducial IGM history.

        Parameters
        ----------
        coeffs, list_coeffs, prop_coeffs, free_param_names
            Fixed histories or metadata used to construct sampled coefficient
            histories.
        z_0 : float, default=3
            Pivot redshift for polynomial histories.
        fid_igm : mapping
            Fiducial quantity and redshift arrays used to build interpolators.
        fid_vals, flat_priors, Gauss_priors : mapping
            Defaults and likelihood-prior metadata for every history family.
        """
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

            self.set_coefficients(free_param_names)

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
        """Fit/interpolate one fiducial IGM history for stable evaluation.

        Parameters
        ----------
        fid_igm : mapping
            Quantity arrays and matching ``<name>_z`` grids.
        name_coeff : str
            Physical IGM quantity name.
        order_extra : int, default=2
            Polynomial order used to smooth the fiducial history.
        smoothing : bool, default=True
            Use the fitted smooth history instead of original nonzero points.
        zmin, zmax : float, default=1.9, 5.5
            Redshift range covered by the resulting interpolator.

        Returns
        -------
        None
            Stores a cubic fiducial interpolator in ``fid_interp``.

        Raises
        ------
        ValueError
            If no finite nonzero fiducial values are available.
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
