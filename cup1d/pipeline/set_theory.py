import numpy as np

from cup1d.likelihood.lya_theory import Theory
from cup1d.likelihood.model_igm import IGM
from cup1d.likelihood.model_contaminants import Contaminants
from cup1d.likelihood.model_systematics import Systematics
from cup1d.likelihood.cosmologies import set_cosmo


def set_theory(
    args, emulator, free_parameters, use_hull=True, fid_or_true="fid", zs=None
):
    """Set theory"""

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

    true_cosmo = set_cosmo(cosmo_label=cosmo_label)
    if zs is None:
        zs = np.concatenate(
            [np.arange(2.2, 4.401, 0.2), np.arange(2.0, 4.501, 0.25)]
        )
    theory.set_fid_cosmo(np.unique(zs), input_cosmo=true_cosmo)

    theory.model_igm.set_fid_igm(zs)

    return theory
