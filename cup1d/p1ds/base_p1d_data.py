import os
import numpy as np

from cup1d.conventions import validate_p1d_contract

from cup1d.utils.utils import get_path_repo


def _drop_zbins(
    z_in,
    k_in,
    Pk_in,
    cov_in,
    z_min,
    z_max,
    full_zs=None,
    full_Pk_kms=None,
    full_cov_kms=None,
    full_cov_stat_kms=None,
    Pksmooth_kms=None,
    cov_stat=None,
    kmin_in=None,
    kmax_in=None,
):
    """Select a redshift interval and remove padded P1D bins.

    Parameters
    ----------
    z_in : array_like of float
        Redshift of each input P1D block.
    k_in, Pk_in : sequence of ndarray
        Per-redshift wavenumber grids in ``s / km`` and P1D measurements in
        ``km / s``.  Zero-valued trailing P1D entries are treated as padding.
    cov_in : sequence of ndarray
        Per-redshift P1D covariance blocks with shape ``(nk, nk)`` and units
        ``(km / s)**2``.
    z_min, z_max : float
        Inclusive redshift limits.
    full_zs, full_Pk_kms, full_cov_kms, full_cov_stat_kms : ndarray, optional
        Optional concatenated representation, restricted consistently with
        the selected redshift range.
    Pksmooth_kms : sequence of ndarray, optional
        Optional smooth P1D prediction associated with each measurement.
    cov_stat : sequence of ndarray, optional
        Optional statistical covariance blocks.
    kmin_in, kmax_in : sequence of ndarray, optional
        Lower and upper wavenumber-bin edges.  When omitted, they are inferred
        from the spacing of each retained grid.

    Returns
    -------
    tuple
        Filtered per-redshift and concatenated P1D arrays, followed by smooth
        P1D, statistical covariance, and bin-edge arrays.  The tuple order is
        the one consumed by :class:`BaseDataP1D`.
    """

    # k_in center of the kbin
    # kmin_in starting of the kbin
    # kmax_in ending of the kbin

    z_in = np.array(z_in)
    ind = np.argwhere((z_in >= z_min) & (z_in <= z_max))[:, 0]
    z_out = z_in[ind]

    k_out = []
    Pk_out = []
    cov_out = []
    cov_stat_out = []
    kmin_out = []
    kmax_out = []
    if Pksmooth_kms is None:
        Pksmooth_out = None
    else:
        Pksmooth_out = []
    if Pksmooth_kms is None:
        Pksmooth_out = None
    for jj in ind:
        # remove tailing zeros
        ind2 = np.argwhere(Pk_in[jj] != 0)[:, 0]
        k_out.append(k_in[jj][ind2])
        if kmin_in is not None:
            kmin_out.append(kmin_in[jj][ind2])
        else:
            kdiff = 0.5 * (k_in[jj][ind2][1] - k_in[jj][ind2][0])
            kmin_out.append(k_in[jj][ind2] - kdiff)
        if kmax_in is not None:
            kmax_out.append(kmax_in[jj][ind2])
        else:
            kdiff = 0.5 * (k_in[jj][ind2][1] - k_in[jj][ind2][0])
            kmax_out.append(k_in[jj][ind2] + kdiff)
        Pk_out.append(Pk_in[jj][ind2])
        if Pksmooth_kms is not None:
            Pksmooth_out.append(Pksmooth_kms[jj][ind2])
        cov_out.append(cov_in[jj][ind2, :][:, ind2])
        if cov_stat is not None:
            cov_stat_out.append(cov_stat[jj][ind2, :][:, ind2])

    if full_zs is not None:
        ind = np.argwhere((full_zs >= z_min) & (full_zs <= z_max))[:, 0]
        full_zs = full_zs[ind]
        full_Pk_kms = full_Pk_kms[ind]
        full_cov_kms = full_cov_kms[ind, :][:, ind]
        if full_cov_stat_kms is not None:
            full_cov_stat_kms = full_cov_stat_kms[ind, :][:, ind]

    return (
        z_out,
        k_out,
        Pk_out,
        cov_out,
        full_zs,
        full_Pk_kms,
        full_cov_kms,
        full_cov_stat_kms,
        Pksmooth_out,
        cov_stat_out,
        kmin_out,
        kmax_out,
    )


class BaseDataP1D(object):
    """Container for binned one-dimensional Lyman-alpha power measurements.

    Instances retain one wavenumber grid and covariance matrix for every
    redshift bin.  The canonical attributes use explicit units: ``k_ikms`` is
    in ``s / km``, ``P1D_kms`` in ``km / s``, and ``cov_P1D_kms`` in
    ``(km / s)**2``.  Legacy ``Pk_kms`` and ``k_kms`` aliases remain available
    for stored analyses.
    """

    BASEDIR = os.path.join(get_path_repo("cup1d"), "data", "p1d_measurements")

    def __init__(
        self,
        z,
        _k_kms,
        Pk_kms,
        cov_Pk_kms,
        z_min=0,
        z_max=10,
        full_zs=None,
        full_Pk_kms=None,
        full_cov_kms=None,
        full_cov_stat_kms=None,
        Pksmooth_kms=None,
        cov_stat=None,
        k_kms_min=None,
        k_kms_max=None,
    ):
        """Initialize a P1D data set and apply its redshift selection.

        Parameters
        ----------
        z : array_like of float
            Redshift-bin centers.
        _k_kms : array_like or sequence of ndarray
            Wavenumber grid(s) in ``s / km``.  A common grid is expanded to
            all redshift bins; otherwise one grid per redshift is required.
        Pk_kms : sequence of ndarray
            Measured P1D values in ``km / s``.
        cov_Pk_kms : sequence of ndarray
            P1D covariance blocks in ``(km / s)**2``.
        z_min, z_max : float, default: 0, 10
            Inclusive redshift bounds retained in this instance.
        full_zs, full_Pk_kms, full_cov_kms, full_cov_stat_kms : ndarray, optional
            Optional concatenated data-vector representation and covariance.
        Pksmooth_kms : sequence of ndarray, optional
            Smooth P1D prediction associated with the measurement.
        cov_stat : sequence of ndarray, optional
            Statistical-only covariance blocks.
        k_kms_min, k_kms_max : sequence of ndarray, optional
            Lower and upper wavenumber-bin edges in ``s / km``.

        Raises
        ------
        ValueError
            If a retained P1D block violates the canonical wavenumber, P1D,
            or covariance shape and unit contract.
        """

        ## if multiple z, ensure that k_kms for each redshift
        # more than one z, and k_kms is different for each z
        if (len(z) > 1) & (len(np.atleast_1d(_k_kms[0])) != 1):
            k_kms = []
            for iz in range(len(z)):
                k_kms.append(_k_kms[iz])
        # more than one z, and kms is the same for all z
        elif (len(z) > 1) & (len(np.atleast_1d(_k_kms[0])) == 1):
            k_kms = []
            for iz in range(len(z)):
                k_kms.append(_k_kms)
        # only one z
        else:
            k_kms = _k_kms

        # drop zbins below z_min and above z_max
        res = _drop_zbins(
            z,
            k_kms,
            Pk_kms,
            cov_Pk_kms,
            z_min,
            z_max,
            full_zs=full_zs,
            full_Pk_kms=full_Pk_kms,
            full_cov_kms=full_cov_kms,
            full_cov_stat_kms=full_cov_stat_kms,
            Pksmooth_kms=Pksmooth_kms,
            cov_stat=cov_stat,
            kmin_in=k_kms_min,
            kmax_in=k_kms_max,
        )

        (
            self.z,
            self.k_ikms,
            self.P1D_kms,
            self.cov_P1D_kms,
            self.full_zs,
            self.full_P1D_kms,
            self.full_cov_P1D_kms,
            self.full_cov_stat_P1D_kms,
            self.P1Dsmooth_kms,
            self.covstat_P1D_kms,
            self.k_ikms_min,
            self.k_ikms_max,
        ) = res

        self.full_k_ikms = np.concatenate(self.k_ikms)
        for k_ikms, P1D_kms, cov_P1D_kms in zip(
            self.k_ikms, self.P1D_kms, self.cov_P1D_kms
        ):
            validate_p1d_contract(k_ikms, P1D_kms, cov_P1D_kms)

        # decide if applying blinding
        self.apply_blinding = False
        if hasattr(self, "blinding"):
            if self.blinding is not None:
                self.apply_blinding = True

    # Compatibility properties for releases and stored analysis code predating
    # the explicit inverse-unit convention.
    k_kms = property(lambda self: self.k_ikms, lambda self, value: setattr(self, "k_ikms", value))
    Pk_kms = property(lambda self: self.P1D_kms, lambda self, value: setattr(self, "P1D_kms", value))
    cov_Pk_kms = property(lambda self: self.cov_P1D_kms, lambda self, value: setattr(self, "cov_P1D_kms", value))
    full_Pk_kms = property(lambda self: self.full_P1D_kms, lambda self, value: setattr(self, "full_P1D_kms", value))
    full_cov_Pk_kms = property(lambda self: self.full_cov_P1D_kms, lambda self, value: setattr(self, "full_cov_P1D_kms", value))
    full_cov_stat_Pk_kms = property(lambda self: self.full_cov_stat_P1D_kms, lambda self, value: setattr(self, "full_cov_stat_P1D_kms", value))
    Pksmooth_kms = property(lambda self: self.P1Dsmooth_kms, lambda self, value: setattr(self, "P1Dsmooth_kms", value))
    covstat_Pk_kms = property(lambda self: self.covstat_P1D_kms, lambda self, value: setattr(self, "covstat_P1D_kms", value))
    k_kms_min = property(lambda self: self.k_ikms_min, lambda self, value: setattr(self, "k_ikms_min", value))
    k_kms_max = property(lambda self: self.k_ikms_max, lambda self, value: setattr(self, "k_ikms_max", value))
    full_k_kms = property(lambda self: self.full_k_ikms, lambda self, value: setattr(self, "full_k_ikms", value))

    def get_P1D_iz(self, iz):
        """Return the P1D vector at one redshift bin.

        Parameters
        ----------
        iz : int
            Index into :attr:`z`.

        Returns
        -------
        ndarray of float
            P1D values with shape ``(nk,)`` and units ``km / s``.
        """
        return self.P1D_kms[iz]

    def get_Pk_iz(self, iz):
        """Return a P1D vector through the legacy ``Pk`` API.

        Parameters
        ----------
        iz : int
            Index into :attr:`z`.

        Returns
        -------
        ndarray of float
            Alias for :meth:`get_P1D_iz` with units ``km / s``.
        """
        return self.get_P1D_iz(iz)

    def get_cov_iz(self, iz):
        """Return the P1D covariance block at one redshift bin.

        Parameters
        ----------
        iz : int
            Index into :attr:`z`.

        Returns
        -------
        ndarray of float
            Covariance matrix with shape ``(nk, nk)`` and units
            ``(km / s)**2``.
        """

        return self.cov_Pk_kms[iz]

    def get_icov_iz(self, iz):
        """Return the inverse P1D covariance block at one redshift bin.

        Parameters
        ----------
        iz : int
            Index into :attr:`z`.

        Returns
        -------
        ndarray of float
            Inverse covariance matrix with shape ``(nk, nk)`` and units
            ``(s / km)**2``.
        """

        return self.icov_Pk_kms[iz]

    def cull_data(self, kmin_kms=0, kmax_kms=10):
        """Restrict every redshift block to an inclusive wavenumber interval.

        Parameters
        ----------
        kmin_kms, kmax_kms : float or None, default: 0, 10
            Lower and upper retained limits in ``s / km``.  Passing ``None``
            for both leaves the data unchanged.

        Notes
        -----
        This method mutates the per-redshift wavenumber, P1D, covariance, and
        inverse-covariance arrays in place.
        """

        if (kmin_kms is None) & (kmax_kms is None):
            return

        for iz in range(len(self.z)):
            ind = np.argwhere(
                (self.k_kms[iz] >= kmin_kms) & (self.k_kms[iz] <= kmax_kms)
            )[:, 0]
            sli = slice(ind[0], ind[-1] + 1)
            self.k_kms[iz] = self.k_kms[iz][sli]
            self.Pk_kms[iz] = self.Pk_kms[iz][sli]
            self.cov_Pk_kms[iz] = self.cov_Pk_kms[iz][sli, sli]
            self.icov_Pk_kms[iz] = self.icov_Pk_kms[iz][sli, sli]

    def plot_p1d(
        self,
        use_dimensionless=True,
        xlog=False,
        ylog=True,
        fname=None,
        cov_ext=None,
        ftsize=18,
        store_data=False,
    ):
        """Plot the measured P1D blocks and their uncertainties.

        Parameters
        ----------
        use_dimensionless : bool, default: True
            Plot dimensionless power when true.
        xlog, ylog : bool, default: False, True
            Use logarithmic axes for wavenumber and power.
        fname : str or path-like, optional
            Output filename passed to the plotting helper.
        cov_ext : sequence of ndarray, optional
            Optional external covariance to display with the data.
        ftsize : float, default: 18
            Base font size for the figure.
        store_data : bool, default: False
            Request storage of the plotted data from the helper.

        Returns
        -------
        object
            Figure or plotting payload returned by
            :func:`cup1d.postprocessing.data.p1d.plot_p1d`.
        """
        from cup1d.postprocessing.data import p1d

        return p1d.plot_p1d(
            self.z,
            self.k_kms,
            self.Pk_kms,
            self.cov_Pk_kms,
            use_dimensionless=use_dimensionless,
            xlog=xlog,
            ylog=ylog,
            fname=fname,
            cov_ext=cov_ext,
            ftsize=ftsize,
            store_data=store_data,
        )
