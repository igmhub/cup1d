Scientific conventions and data contracts
=========================================

Names and units
---------------

Public dimensional names contain their units, and ``i`` denotes an inverse
unit. Thus ``k_iMpc`` is in Mpc^-1 and ``k_ikms`` is in
(km/s)^-1 = s/km. Power names are always capitalized:
``P1D_Mpc`` (Mpc), ``P3D_Mpc`` (Mpc^3), and
``P1D_kms`` (km/s). The conversion ``dkms_diMpc`` has units
(km/s)/Mpc.

The theory boundary converts according to
``k_iMpc = k_ikms * dkms_diMpc`` and
``P1D_kms = P1D_Mpc * dkms_diMpc``. Dimensionless likelihood and
cosmological parameters do not receive a unit suffix.

Data and likelihood arrays
--------------------------

For each redshift bin ``iz``, ``data.k_ikms[iz]`` and
``data.P1D_kms[iz]`` are one-dimensional arrays of the same length.
``data.cov_P1D_kms[iz]`` has shape ``(Nk, Nk)`` and units
(km/s)^2. The corresponding ``full_*`` arrays concatenate bins in the
same order used by the likelihood covariance. Emulator calls use an
Mpc-space wavenumber array with redshift/model on the first axis and
wavenumber on the last.

Emulator inputs, ranges, and uncertainty
----------------------------------------

The emulator's ``emu_params`` sequence is the authoritative input order
when an ordered representation is required; mappings are used otherwise.
``kmax_iMpc`` is the validated upper wavenumber. Values outside emulator
training bounds are extrapolations. Covariance results always describe the
returned P1D vector, have shape ``(N, N)``, and carry squared P1D units;
standard-deviation bands carry P1D units.

Compatibility
-------------

Old attributes such as ``k_kms``, ``Pk_kms``,
``cov_Pk_kms``, ``kmax_Mpc``, and methods named
``get_p1d_kms`` remain aliases during migration. Existing YAML and
archive products remain loadable. New code should use only the canonical
spellings.
