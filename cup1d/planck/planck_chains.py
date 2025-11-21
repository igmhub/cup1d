import os
from getdist import loadMCSamples
import cup1d


def planck_chains_dir(release, root_dir):
    """Given a Planck data release (year, integer), return the full path
    to the folder where the chains are stored.
    If no root_dir is passed, use environmental variable PLANCK_CHAINS."""

    if root_dir is None:
        root_dir = os.path.join(
            os.path.dirname(cup1d.__path__[0]),
            "data",
            "planck_linP_chains",
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
    """Check that input chain exist, at least in zipped format, and read them."""

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

    return samples


def get_planck_results(release, model, data, root_dir, linP_tag):
    """Load results from Planck, for a given data release and data combination.
    Inputs:
        - release (integer): 2013, 2015 or 2018
        - model (string): cosmo model, e.g., base, base_mnu...
        - data (string): data combination, e.g., plikHM_TT_lowl_lowE
        - root_dir (string): path to folder with Planck chains
        - linP_tag (string): label identifying linear power columns
    Outputs:
        - dictionary with relevant information
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
    """Load results from Planck 2013 chain"""
    return get_planck_results(
        2013, model=model, data=data, root_dir=root_dir, linP_tag=linP_tag
    )


def get_planck_2015(
    model="base_mnu", data="plikHM_TT_lowTEB", root_dir=None, linP_tag="zlinP"
):
    """Load results from Planck 2015 chain"""
    return get_planck_results(
        2015, model=model, data=data, root_dir=root_dir, linP_tag=linP_tag
    )


def get_planck_2018(
    model="base_mnu",
    data="plikHM_TTTEEE_lowl_lowE",
    root_dir=None,
    linP_tag="zlinP",
):
    """Load results from Planck 2018 chain.
    - linP_tag identifies chains with added linear parameters."""
    return get_planck_results(
        2018, model=model, data=data, root_dir=root_dir, linP_tag=linP_tag
    )


def get_cobaya(
    root_dir=None,
    model="base_mnu",
    data="DESI_CMB-SPA",
    linP_tag="zlinP",
):
    """Load results from Planck, for a given data release and data combination.
    Inputs:
        - release (integer): 2013, 2015 or 2018
        - model (string): cosmo model, e.g., base, base_mnu...
        - data (string): data combination, e.g., plikHM_TT_lowl_lowE
        - root_dir (string): path to folder with Planck chains
        - linP_tag (string): label identifying linear power columns
    Outputs:
        - dictionary with relevant information
    """

    from cobaya.yaml import yaml_load_file
    from cobaya import load_samples

    if linP_tag is not None:
        folder = os.path.join(root_dir, model, data, linP_tag + "/")
    else:
        folder = os.path.join(root_dir, model, data + "/")

    name_chain = model + "_" + data
    info_from_yaml = yaml_load_file(folder + name_chain + ".input.yaml")
    info_from_yaml["output"] = folder + name_chain

    gd_sample = load_samples(info_from_yaml["output"], to_getdist=True)

    analysis = {}
    analysis["samples"] = gd_sample
    # analysis["samples"] = load_samples(analysis["dir_name"] + "chain.1.txt")
    analysis["parameters"] = analysis["samples"].getParams()

    return analysis
