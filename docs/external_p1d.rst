External uncontaminated P1D likelihood
============================================================

``cup1d.likelihood.external.ExternalP1DLikelihood`` evaluates the full data
likelihood from externally supplied uncontaminated velocity-space P1D. It does
not load an emulator, run CAMB or native priors, or invoke ``Analysis``. Construct
it with selected data, native contaminant/systematics models, explicit covariance
scaling and trusted error assets, and a frozen fiducial velocity conversion.

.. code-block:: python

   request = backend.get_prediction_request()
   observed = backend.apply_observation_model(powers, context, nuisance_values)
   result = backend.evaluate_from_p1d(powers, context, nuisance_values)

``PredictionRequest`` is frozen and serializable via ``to_dict`` / ``from_dict``.
Groups preserve redshift order and ragged fine k grids (s/km), before contamination
and rebinning. Its hash includes grids, ordering, units, stage and the caller's
configuration identity. ``PredictionContext`` contains that hash, unique
redshifts, current mean flux and current ``dkms_diMpc``. ``powers`` maps group
identifiers to lists of 1D P1D arrays in km/s on exactly those grids. Nuisance
values are native named coefficients, not sampler cubes.

The shared scalar observation helper preserves native ordering:

.. code-block:: text

   (C_HCD * C_mul_metals * IC_corr * P1D_lya + C_add_metals) * C_res

Then native response/rebinning weights apply. Full data covariance, including
cross-redshift blocks, is retained. Covariance construction is shared with native
``Likelihood.set_icov``, which now also accepts explicit ``emulator_covariance``
and ``fiducial_conversion``. Its no-argument behavior is unchanged. The fiducial
conversion must be static and must not consult a changing provider.

The structured result exposes ``loglike``, raw ``chi2_data``, included
``logdet_cov``, ``ndata`` and ``valid``. Only ``ndata*log(2*pi)`` is always omitted;
inclusion of the fixed determinant is selected at initialization. No statistical
prior is added. Select the complete intended dataset first; diagnostic z masks
are not accepted. Malformed predictions, covariance and ordering errors remain
visible. Scientific model-domain exclusions belong to the external theory.

The Cobaya adapter lives in sibling ``lya_interface`` and refuses implicitly
blinded exports of provider-derived cosmological diagnostics.
