import numpy as np
from cup1d.utils.various_dicts import conv_strings


def set_blinding(apply_blinding, seed):
    """Set the blinding parameters"""
    blind_prior = {"Delta2_star": 0.05, "n_star": 0.01, "alpha_star": 0.005}
    if apply_blinding:
        rng = np.random.default_rng(seed)
    blind = {}
    for key in blind_prior:
        if apply_blinding:
            blind[key] = rng.normal(0, blind_prior[key])
        else:
            blind[key] = 0
    return blind


def apply_blinding(blind, dict_cosmo):
    """Apply blinding to the dict_cosmo"""

    for key in blind:

        try:
            dict_cosmo[key] += blind[key]
        except:
            pass

        key2 = conv_strings[key]
        try:
            dict_cosmo[key2] += blind[key]
        except:
            pass

    return dict_cosmo


def apply_unblinding(blind, dict_cosmo):
    """Apply blinding to the dict_cosmo"""

    for key in blind:

        try:
            dict_cosmo[key] -= blind[key]
        except:
            pass

        key2 = conv_strings[key]
        try:
            dict_cosmo[key2] -= blind[key]
        except:
            pass

    return dict_cosmo
