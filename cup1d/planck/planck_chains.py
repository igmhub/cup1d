"""Load Planck, CMB-SPA, and Cobaya chains as GetDist samples."""

import os
import subprocess

from getdist import loadMCSamples

from cup1d.utils.utils import get_path_repo


def spa_chains_dir(root_dir):
    """Return the root directory that stores CMB-SPA linear-power chains."""
    if root_dir is None:
        root_dir = os.path.join(
            get_path_repo("cup1d"), "data", "cmbspa_linP_chains"
        )
    print("root_dir", root_dir)
    return root_dir


def planck_chains_dir(release, root_dir):
    """Return the chain directory for a Planck release."""

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
    """Load a GetDist chain, unzipping ``.txt.gz`` chain files if needed."""

    print("loading", file_root)

    try:
        samples = loadMCSamples(file_root)
    except OSError:
        if os.path.exists(file_root + ".txt.gz"):
            print("unzip chain", file_root)
            subprocess.run(["gzip", "-dk", file_root + ".txt.gz"], check=True)
            samples = loadMCSamples(file_root)
        else:
            raise OSError("No chains found (not even zipped): " + file_root) from None

    return samples


def get_planck_results(release, model, data, root_dir, linP_tag):
    """Load Planck chains for one release, model, and data combination."""

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
    """Load a Planck 2013 chain."""
    return get_planck_results(
        2013, model=model, data=data, root_dir=root_dir, linP_tag=linP_tag
    )


def get_planck_2015(
    model="base_mnu", data="plikHM_TT_lowTEB", root_dir=None, linP_tag="zlinP"
):
    """Load a Planck 2015 chain."""
    return get_planck_results(
        2015, model=model, data=data, root_dir=root_dir, linP_tag=linP_tag
    )


def get_planck_2018(
    model="base_mnu",
    data="plikHM_TTTEEE_lowl_lowE",
    root_dir=None,
    linP_tag="zlinP",
):
    """Load a Planck 2018 chain."""
    return get_planck_results(
        2018, model=model, data=data, root_dir=root_dir, linP_tag=linP_tag
    )


def get_spa_results(model, data, root_dir, linP_tag, release="d1"):
    """Load CMB-SPA chains for one model and data combination."""

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
    """Load the default CMB-SPA chain."""
    return get_spa_results(model, data, root_dir, linP_tag)


def get_cobaya(
    root_dir=None,
    model="base_mnu",
    data="DESI_CMB-SPA",
    linP_tag="zlinP",
    lite=False,
):
    """Load a Cobaya chain and convert it to GetDist samples."""

    from cobaya import load_samples
    from cobaya.yaml import yaml_load_file

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
