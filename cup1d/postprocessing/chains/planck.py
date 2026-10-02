import os
from collections.abc import Iterable

from getdist import loadMCSamples
from cup1d.utils.utils import get_path_repo


def spa_chains_dir(root_dir):
    """Resolve the directory containing bundled CMB-SPA chains.

    Parameters
    ----------
    root_dir : str or path-like or None
        Explicit chain root.  When None, use Cup1D's bundled SPA-chain data.

    Returns
    -------
    str
        Chain-root directory path.
    """
    if root_dir is None:
        root_dir = os.path.join(
            get_path_repo("cup1d"), "data", "cmbspa_linP_chains"
        )
    print("root_dir", root_dir)
    return root_dir


def planck_chains_dir(release, root_dir):
    """Resolve the directory for a bundled Planck release.

    Parameters
    ----------
    release : {2013, 2015, 2018}
        Planck release year.
    root_dir : str or path-like or None
        Parent directory holding Planck chain releases.  When None, use the
        bundled Cup1D chain directory.

    Returns
    -------
    str
        Release-specific chain directory.

    Raises
    ------
    ValueError
        If ``release`` is unsupported.
    """

    if root_dir is None:
        root_dir = os.path.join(
            get_path_repo("cup1d"), "data", "planck_linP_chains"
        )
    print("root_dir", root_dir)
    if release == 2013:
        return os.path.join(root_dir, "COM_CosmoParams_fullGrid_R1.10")
    elif release == 2015:
        return os.path.join(root_dir, "COM_CosmoParams_fullGrid_R2.00")
    elif release == 2018:
        return os.path.join(root_dir, "COM_CosmoParams_fullGrid_R3.01")
    else:
        raise ValueError("wrong Planck release", release)


def load_samples(file_root):
    """Load a GetDist chain, unpacking a gzipped text chain when necessary.

    Parameters
    ----------
    file_root : str or path-like
        GetDist file root, without the optional ``.txt`` or ``.txt.gz`` suffix.

    Returns
    -------
    getdist.mcsamples.MCSamples
        Loaded samples, with a ``periodic`` range attribute supplied for
        legacy Planck range files.

    Raises
    ------
    IOError
        If neither an unpacked nor gzipped chain is available.
    """

    print("loading", file_root)

    try:
        samples = loadMCSamples(file_root)
    except IOError:
        if os.path.exists(file_root + ".txt.gz"):
            print("unzip chain", file_root)
            cmd = "gzip -dk " + file_root + ".txt.gz"
            os.system(cmd)
            samples = loadMCSamples(file_root)
        else:
            raise IOError("No chains found (not even zipped): " + file_root)

    # Legacy Planck range files predate this GetDist attribute. Current
    # GetDist accesses it while producing marginal distributions.
    if not hasattr(samples.ranges, "periodic"):
        samples.ranges.periodic = set()
    return samples


def get_planck_results(release, model, data, root_dir, linP_tag):
    """Load one Planck chain and its derived parameter accessor.

    Parameters
    ----------
    release : {2013, 2015, 2018}
        Planck release year.
    model, data : str
        Cosmological model and Planck likelihood-combination directory names.
    root_dir : str or path-like or None
        Parent directory containing Planck chains.
    linP_tag : str or None
        Suffix identifying added linear-power parameter columns.  None selects
        the chain without that suffix.

    Returns
    -------
    dict
        Mapping containing release metadata, resolved paths, chain name,
        :class:`getdist.mcsamples.MCSamples`, and its parameter accessor.
    """

    analysis = {}
    analysis["release"] = release
    analysis["release_dir"] = planck_chains_dir(
        release=release, root_dir=root_dir
    )
    # specify analysis and chain name
    analysis["model"] = model
    analysis["data"] = data
    analysis["dir_name"] = (
        analysis["release_dir"]
        + "/"
        + analysis["model"]
        + "/"
        + analysis["data"]
        + "/"
    )
    # specify linear power parameters added (if any)
    analysis["linP_tag"] = linP_tag
    if linP_tag is None:
        analysis["chain_name"] = analysis["model"] + "_" + analysis["data"]
    else:
        analysis["chain_name"] = (
            analysis["model"]
            + "_"
            + analysis["data"]
            + "_"
            + analysis["linP_tag"]
        )
    # load and store chains read from file
    analysis["samples"] = load_samples(
        analysis["dir_name"] + analysis["chain_name"]
    )
    analysis["parameters"] = analysis["samples"].getParams()

    return analysis


def get_planck_2013(
    model="base_mnu",
    data="planck_lowl_lowLike_highL",
    root_dir=None,
    linP_tag="zlinP",
):
    """Load a Planck-2013 chain.

    Parameters
    ----------
    model, data, root_dir, linP_tag
        Forwarded to :func:`get_planck_results` with release 2013.

    Returns
    -------
    dict
        Loaded-chain metadata and GetDist objects.
    """
    return get_planck_results(
        2013, model=model, data=data, root_dir=root_dir, linP_tag=linP_tag
    )


def get_planck_2015(
    model="base_mnu", data="plikHM_TT_lowTEB", root_dir=None, linP_tag="zlinP"
):
    """Load a Planck-2015 chain.

    Parameters
    ----------
    model, data, root_dir, linP_tag
        Forwarded to :func:`get_planck_results` with release 2015.

    Returns
    -------
    dict
        Loaded-chain metadata and GetDist objects.
    """
    return get_planck_results(
        2015, model=model, data=data, root_dir=root_dir, linP_tag=linP_tag
    )


def get_planck_2018(
    model="base_mnu",
    data="plikHM_TTTEEE_lowl_lowE",
    root_dir=None,
    linP_tag="zlinP",
):
    """Load a Planck-2018 chain.

    Parameters
    ----------
    model, data, root_dir, linP_tag
        Forwarded to :func:`get_planck_results` with release 2018.

    Returns
    -------
    dict
        Loaded-chain metadata and GetDist objects.
    """
    return get_planck_results(
        2018, model=model, data=data, root_dir=root_dir, linP_tag=linP_tag
    )


def load_planck_2018_chains(chain_specs: Iterable[dict], root_dir=None):
    """Load a named collection of Planck-2018 chains.

    Parameters
    ----------
    chain_specs : iterable of dict
        Specifications requiring ``model`` and ``data``.  ``name`` defaults to
        ``model`` and ``linP_tag`` defaults to None; extra plotting metadata is
        ignored.
    root_dir : str or path-like or None, optional
        Parent directory containing Planck chains.

    Returns
    -------
    dict[str, dict]
        Loaded chain metadata keyed by each requested name.

    Raises
    ------
    ValueError
        If two specifications resolve to the same output name.
    """
    chains = {}
    for spec in chain_specs:
        name = spec.get("name", spec["model"])
        if name in chains:
            raise ValueError(f"Duplicate Planck chain name: {name}")
        chains[name] = get_planck_2018(
            model=spec["model"],
            data=spec["data"],
            root_dir=root_dir,
            linP_tag=spec.get("linP_tag"),
        )
    return chains


def load_spa_chains(chain_specs: Iterable[dict], root_dir=None):
    """Load a named collection of CMB-SPA chains.

    Parameters
    ----------
    chain_specs : iterable of dict
        Specifications following :func:`load_planck_2018_chains`; ``linP_tag``
        defaults to the CMB-SPA convention, ``"linP"``.
    root_dir : str or path-like or None, optional
        Parent directory containing CMB-SPA chains.

    Returns
    -------
    dict[str, dict]
        Loaded chain metadata keyed by each requested name.

    Raises
    ------
    ValueError
        If two specifications resolve to the same output name.
    """
    chains = {}
    for spec in chain_specs:
        name = spec.get("name", spec["model"])
        if name in chains:
            raise ValueError(f"Duplicate CMB-SPA chain name: {name}")
        chains[name] = get_spa(
            model=spec["model"],
            data=spec["data"],
            root_dir=root_dir,
            linP_tag=spec.get("linP_tag", "linP"),
        )
    return chains


def get_spa_results(model, data, root_dir, linP_tag, release="d1"):
    """Load one CMB-SPA chain and its parameter accessor.

    Parameters
    ----------
    model, data : str
        Cosmological model and data-combination directory names.
    root_dir : str or path-like or None
        Parent directory containing CMB-SPA chains.
    linP_tag : str or None
        Suffix identifying linear-power augmented chains.  None chooses the
        untagged directory and chain name.
    release : str, default: "d1"
        CMB-SPA release identifier stored in the returned metadata.

    Returns
    -------
    dict
        Resolved chain metadata, loaded GetDist samples, and parameter accessor.
    """

    analysis = {}
    analysis["release"] = release
    analysis["release_dir"] = spa_chains_dir(root_dir=root_dir) + "/"
    # specify analysis and chain name
    analysis["model"] = model
    analysis["data"] = data
    analysis["dir_name"] = (
        analysis["release_dir"] + analysis["model"] + "/" + analysis["data"]
    )
    # specify linear power parameters added (if any)
    analysis["linP_tag"] = linP_tag

    if linP_tag is None:
        analysis["dir_name"] += "/"
        analysis["chain_name"] = analysis["model"] + "_" + analysis["data"]
    else:
        analysis["dir_name"] += "_" + linP_tag + "/"
        analysis["chain_name"] = (
            analysis["model"]
            + "_"
            + analysis["data"]
            + "_"
            + analysis["linP_tag"]
        )
    # load and store chains read from file
    analysis["samples"] = load_samples(
        analysis["dir_name"] + analysis["chain_name"]
    )
    analysis["parameters"] = analysis["samples"].getParams()

    return analysis


def get_spa(
    model="base_mnu", data="DESI_CMB-SPA", root_dir=None, linP_tag="linP"
):
    """Load a standard CMB-SPA chain.

    Parameters
    ----------
    model, data, root_dir, linP_tag
        Forwarded to :func:`get_spa_results`.

    Returns
    -------
    dict
        Loaded-chain metadata and GetDist objects.
    """
    return get_spa_results(model, data, root_dir, linP_tag)


def get_cobaya(
    root_dir=None,
    model="base_mnu",
    data="DESI_CMB-SPA",
    linP_tag="zlinP",
    lite=False,
):
    """Load a Cobaya output chain as GetDist samples.

    Parameters
    ----------
    root_dir : str or path-like
        Parent directory containing the Cobaya model and data directories.
    model, data : str
        Cosmological model and data-combination directory names.
    linP_tag : str or None, default: "zlinP"
        Optional subdirectory identifying added linear-power parameters.
    lite : bool, default: False
        Load the ``-lite`` output root when true.

    Returns
    -------
    dict
        Mapping with loaded GetDist samples and its parameter accessor.
    """

    from cobaya.yaml import yaml_load_file
    from cobaya import load_samples

    if linP_tag is not None:
        folder = os.path.join(root_dir, model, data, linP_tag + "/")
    else:
        folder = os.path.join(root_dir, model, data + "/")

    name_chain = model + "_" + data
    if lite:
        name_chain += "-lite"
    info_from_yaml = yaml_load_file(folder + name_chain + ".input.yaml")
    info_from_yaml["output"] = folder + name_chain

    gd_sample = load_samples(info_from_yaml["output"], to_getdist=True)

    analysis = {}
    analysis["samples"] = gd_sample
    # analysis["samples"] = load_samples(analysis["dir_name"] + "chain.1.txt")
    analysis["parameters"] = analysis["samples"].getParams()

    return analysis
