import numpy as np
from lace.cosmo import cosmology

from cup1d.models.contaminants.model_contaminants import Contaminants
from cup1d.models.contaminants.model_systematics import Systematics
from cup1d.models.igm.model_igm import IGM
from cup1d.theory.theory import Theory


def set_theory(
    args,
    emulator,
    free_parameters,
    use_hull=True,
    fid_or_true="fid",
    zs=None,
):
    """Construct a configured theory model for fiducial or synthetic data.

    Parameters
    ----------
    args : cup1d.configuration.args.Args
        Analysis configuration supplying fiducial/true models and priors.
    emulator : object
        P1D emulator compatible with :class:`cup1d.theory.theory.Theory`.
    free_parameters : sequence of str
        Parameter names enabled in IGM, contaminant, and systematic models.
    use_hull : bool, default: True
        Enable emulator-domain hull checks in the constructed theory.
    fid_or_true : {"fid", "true"}, default: "fid"
        Select fiducial or synthetic-truth model settings.
    zs : array_like, optional
        Redshifts used for cosmology and fiducial IGM initialization.

    Returns
    -------
    cup1d.theory.theory.Theory
        Initialized theory model.

    Raises
    ------
    ValueError
        If ``fid_or_true`` is neither ``"fid"`` nor ``"true"``.
    """

    if zs is None:
        zs = np.arange(2.0, 4.51, 0.1)

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

    model_igm = IGM(free_param_names=free_parameters, pars_igm=pars_igm)

    model_cont = Contaminants(
        free_param_names=free_parameters,
        pars_cont=pars_cont,
        ic_correction=args.ic_correction,
    )

    model_syst = Systematics(free_param_names=free_parameters, pars_syst=pars_syst)

    theory = Theory(
        emulator=emulator,
        model_igm=model_igm,
        model_cont=model_cont,
        model_syst=model_syst,
        use_hull=use_hull,
        use_star_priors=args.use_star_priors,
        z_star=args.z_star,
        kp_kms=args.kp_kms,
        cosmo_priors=args.cosmo_priors,
    )
    theory.set_fid_cosmo(np.unique(zs), cosmo_label=cosmo_label)

    theory.model_igm.set_fid_igm(np.unique(zs))

    if emulator.emulator_label in ("forest_mpg", "forest_mpg_fix"):
        emulator_cosmology = cosmology.Cosmology(cosmo_label=cosmo_label)
        emulator.set_cosmology(emulator_cosmology.input_cosmo_params_dict)

    return theory
