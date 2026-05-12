"""Factory for likelihood theory objects."""

import numpy as np

from cup1d.likelihood.lya_theory import Theory
from cup1d.likelihood.model_igm import IGM
from cup1d.likelihood.model_contaminants import Contaminants
from cup1d.likelihood.model_systematics import Systematics
from cup1d.likelihood.cosmologies import set_cosmo


def set_theory(
    args,
    emulator,
    free_parameters,
    use_hull=True,
    fid_or_true="fid",
    zs=None,
):
    """Build the theory object used by the likelihood pipeline.

    Parameters
    ----------
    args : cup1d.likelihood.input_pipeline.Args
        Pipeline configuration containing fiducial/true model settings.
    emulator : object
        P1D emulator used by :class:`cup1d.likelihood.lya_theory.Theory`.
    free_parameters : list[str]
        Likelihood parameter names that should be varied.
    use_hull : bool, optional
        Whether to enforce emulator convex-hull checks.
    fid_or_true : {"fid", "true"}, optional
        Select fiducial or true model dictionaries from ``args``.
    zs : array-like or None, optional
        Redshift grid used to initialize fiducial cosmology and IGM values.
    """

    if fid_or_true == "fid":
        pars_igm = args.fid_igm
        pars_cont = args.fid_cont
        pars_syst = args.fid_syst
        cosmo_label = args.fid_cosmo_label
    elif fid_or_true == "true":
        pars_igm = args.true_igm
        pars_cont = args.true_cont
        pars_syst = args.true_syst
        cosmo_label = args.true_cosmo_label
    else:
        raise ValueError("fid_or_true must be 'fid' or 'true'")

    # set igm model
    model_igm = IGM(free_param_names=free_parameters, pars_igm=pars_igm)

    # set contaminants
    model_cont = Contaminants(
        free_param_names=free_parameters,
        pars_cont=pars_cont,
        ic_correction=args.ic_correction,
    )

    # set systematics
    model_syst = Systematics(
        free_param_names=free_parameters, pars_syst=pars_syst
    )

    # set theory
    theory = Theory(
        emulator=emulator,
        model_igm=model_igm,
        model_cont=model_cont,
        model_syst=model_syst,
        use_hull=use_hull,
        use_star_priors=args.use_star_priors,
        z_star=args.z_star,
        kp_kms=args.kp_kms,
    )

    true_cosmo = set_cosmo(
        cosmo_label=cosmo_label, nyx_version=args.nyx_training_set
    )
    if zs is None:
        zs = np.concatenate(
            [np.arange(2.2, 4.401, 0.2), np.arange(2.0, 4.501, 0.25)]
        )
    theory.set_fid_cosmo(np.unique(zs), input_cosmo=true_cosmo)

    theory.model_igm.set_fid_igm(zs)

    return theory
