import numpy as np
import pickle
import matplotlib.pyplot as plt


def model_SB1(k, A, gamma, B, b, k1, phi1, C, c, k2, phi2):
    print(B, b, C, c)
    power_law = (
        A * k ** (-gamma)
        + B * np.exp(-b * k) * np.cos(2 * np.pi * ((k - k1) / k1) + phi1)
        + C * np.exp(-c * k) * np.cos(2 * np.pi * ((k - k2) / k2) + phi2)
    )
    return power_law


def model_SB1_indiv_kms(k_kms, param_mean, A, B):
    mean_p = model_SB1(k_kms, *param_mean)
    return (A * k_kms + B) * mean_p


# def subtract_metal(pk, z, file_metal, velunits=True):
#     P_metal_m = prepare_metal_subtraction(zbins, file_metal, velunits=velunits)
#     for z in zbins:
#         pk.p[z] = pk.p[z] - P_metal_m[z](pk.k[z])


def SB1_power(zs, k_kms, file_metal):
    (param_SB1_mean, param_SB1_indiv) = pickle.load(open(file_metal, "rb"))
    P_metal_m = []
    for iz in range(len(zs)):
        _P_metal_m = model_SB1_indiv_kms(
            k_kms[iz],
            param_SB1_mean,
            param_SB1_indiv[iz][0],
            param_SB1_indiv[iz][1],
        )
        P_metal_m.append(_P_metal_m)
    return P_metal_m
