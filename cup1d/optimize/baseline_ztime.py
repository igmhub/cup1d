import numpy as np
from cup1d.likelihood.pipeline import set_like
from cup1d.likelihood.fitter import Fitter
from scipy.stats.distributions import chi2 as chi2_scipy


# list_props = like.free_param_names.copy()
# def chi2_param_at_time(args, list_props):
#     """Add parameter at a time, old"""
#     args.emu_cov_factor = 1
#     args.emu_cov_type = "block"
#     args.rebin_k = 8
#     args.cov_factor = 1
#     args.fix_cosmo = True
#     args.vary_alphas = False
#     args.fid_cosmo_label = "Planck18"
#     sim_fid = "mpg_central"
#     args.fid_label_mF = sim_fid
#     args.fid_label_T = sim_fid
#     args.fid_label_kF = sim_fid

#     lines_use = [
#         "Lya_SiIII",
#         "Lya_SiII",
#         "SiIIa_SiIII",
#         "SiIIb_SiIII",
#         "SiIIa_SiIIb",
#     ]
#     args.hcd_model_type = "new"
#     args.resolution_model_type = "pivot"
#     args.fid_A_scale = [0, 5]

#     # at a time
#     f_prior = {
#         "Lya_SiIII": -4.2,
#         "Lya_SiIIa": -4.6,
#         "Lya_SiIIb": -4.6,
#         "SiIIa_SiIIb": -5.5,
#         "SiIIa_SiIII": -6.2,
#         "SiIIb_SiIII": -6.6,
#     }

#     # at a time
#     d_prior = {
#         "Lya_SiIII": 0.4,
#         "Lya_SiIIa": -0.9,
#         "Lya_SiIIb": -0.9,
#         "SiIIa_SiIIb": 0.8,
#         "SiIIa_SiIII": 1.6,
#         "SiIIb_SiIII": 2.7,
#     }

#     # at a time
#     a_prior = {
#         "Lya_SiIII": 1.5,
#         "Lya_SiIIa": 4.0,
#         "Lya_SiIIb": 4.0,
#         "SiIIa_SiIIb": 4.0,
#         "SiIIa_SiIII": 0.5,
#         "SiIIb_SiIII": 2.5,
#     }

#     for prop in list_props:
#         if "ln_tau" in prop:
#             args.n_tau = 0
#         else:
#             args.n_tau = 1

#         if "ln_sigT" in prop:
#             args.n_sigT = 0
#         else:
#             args.n_sigT = 1

#         if "ln_gamma" in prop:
#             args.n_gamma = 0
#         else:
#             args.n_gamma = 1

#         if "ln_kF" in prop:
#             args.n_kF = 0
#         else:
#             args.n_kF = 1

#         for metal_line in lines_use:
#             args.fid_metals[metal_line + "_L"] = [0, 0]
#             args.n_metals["n_l_" + metal_line] = 0

#             if "ln_x_" + metal_line in prop:
#                 args.n_metals["n_x_" + metal_line] = 0
#                 args.fid_metals[metal_line + "_X"] = [0, -10.5]
#             else:
#                 args.n_metals["n_x_" + metal_line] = 1
#                 args.fid_metals[metal_line + "_X"] = [0, f_prior[metal_line]]

#             if "d_" + metal_line in prop:
#                 args.n_metals["n_d_" + metal_line] = 0
#                 args.fid_metals[metal_line + "_D"] = [0, 0]
#             else:
#                 args.n_metals["n_d_" + metal_line] = 1
#                 args.fid_metals[metal_line + "_D"] = [0, d_prior[metal_line]]

#             if "a_" + metal_line in prop:
#                 args.n_metals["n_a_" + metal_line] = 0
#                 args.fid_metals[metal_line + "_A"] = [0, 2]
#             else:
#                 args.n_metals["n_a_" + metal_line] = 1
#                 args.fid_metals[metal_line + "_A"] = [0, a_prior[metal_line]]

#         if "R_coeff" in prop:
#             args.n_res = 0
#         else:
#             args.n_res = 1

#         if "ln_A_damp" in prop:
#             args.n_d_dla = 0
#             args.fid_A_damp = [0, -9.5]
#         else:
#             args.n_d_dla = 1
#             args.fid_A_damp = [0, -1.5]

#         if "ln_A_scale" in prop:
#             args.n_s_dla = 0
#             args.fid_A_scale = [0, 5]
#         else:
#             args.n_s_dla = 1
#             args.fid_A_scale = [0, 5.6]

#         args.fid_AGN = [0, -5.5]

#         like = set_like(
#             data["P1Ds"],
#             emulator,
#             args,
#             data_hires=data["extra_P1Ds"],
#         )

#         # print()
#         # f_space_len = 14
#         # s_space_len = 5
#         # for p in like.free_params:
#         #     print(
#         #         p.name,
#         #         (f_space_len - len(p.name)) * " ",
#         #         "\t",
#         #         np.round(p.value, 3),
#         #         (s_space_len - len(str(np.round(p.value, 3)))) * " ",
#         #         "\t",
#         #         np.round(p.min_value, 3),
#         #         (s_space_len - len(str(np.round(p.min_value, 3)))) * " ",
#         #         "\t",
#         #         np.round(p.max_value, 3),
#         #         (s_space_len - len(str(np.round(p.max_value, 3)))) * " ",
#         #         "\t",
#         #         p.Gauss_priors_width,
#         #     )
#         # print()

#         fitter = Fitter(
#             like=like,
#             rootdir=output_dir,
#             nburnin=args.n_burn_in,
#             nsteps=args.n_steps,
#             parallel=args.parallel,
#             explore=args.explore,
#             fix_cosmology=args.fix_cosmo,
#         )

#         p0 = np.array(list(like.fid["fit_cube"].values()))
#         out_mle = []
#         out_mle_cube = []
#         out_chi2 = []
#         for ii in range(len(like.data.z)):
#             # for ii in range(2,3):
#             print(prop, like.data.z[ii])
#             zmask = np.array([like.data.z[ii]])
#             # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
#             # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0, zmask=zmask, restart=True)
#             fitter.run_minimizer(
#                 log_func_minimize=fitter.like.minus_log_prob,
#                 p0=p0,
#                 zmask=zmask,
#                 restart=True,
#                 nsamples=8,
#             )
#             # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
#             out_mle.append(fitter.mle)
#             out_mle_cube.append(fitter.mle_cube)
#             out_chi2.append(fitter.mle_chi2)

#         out = {}
#         out["param_names"] = list_props
#         out["zs"] = like.data.z
#         out["mle"] = out_mle
#         out["mle_cube"] = out_mle_cube
#         out["chi2"] = out_chi2

#         np.save("qmle3_lpo/" + prop + ".npy", out)


def chi2_grow_model_atz(
    folder, args, iz, fix_props, basic_props, label_fit="basic"
):
    """Add parameter at a time, save to disk"""
    fid_vals_metals = {
        "f_Lya_SiIII": -4.0,
        "f_Lya_SiII": -4.0,
        "f_SiIIa_SiIII": 1.0,
        "f_SiIIb_SiIII": 1.0,
        "f_SiIIa_SiIIb": -0.5,
    }

    igm_params = [
        "n_tau",
        "n_gamma",
        "n_sigT",
        "n_kF",
    ]

    list_all_props = [
        "n_tau",
        "n_gamma",
        "n_sigT",
        "n_kF",
        "n_f_Lya_SiIII",
        "n_s_Lya_SiIII",
        "n_f_Lya_SiII",
        "n_s_Lya_SiII",
        "n_f_SiIIa_SiIII",
        "n_f_SiIIb_SiIII",
        "n_f_SiIIa_SiIIb",
        "n_s_SiIIa_SiIIb",
        "n_d_dla1",
        "n_d_dla2",
        "n_d_dla3",
        "n_d_dla4",
    ]

    args.set_baseline()

    like = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )

    out = {}
    out["zs"] = like.data.z[iz]
    out["param_names"] = []
    out["mle"] = []
    out["mle_cube"] = []
    out["chi2"] = []
    out["ndeg"] = []

    for iq, prop in enumerate(basic_props):
        list_props = []

        # setting to zero all props
        for prop2 in list_all_props:
            if prop2 in igm_params:
                args.fid_igm[prop2] = 0
            else:
                args.fid_cont[prop2] = 0

        # setting to one fix props
        for prop2 in fix_props:
            if prop2 in igm_params:
                args.fid_igm[prop2] = 1
            else:
                args.fid_cont[prop2] = 1
            list_props.append(prop2)

        # setting to one basic_props
        if prop is not None:
            if prop in igm_params:
                args.fid_igm[prop] = 1
            else:
                args.fid_cont[prop] = 1
            list_props.append(prop)

        for metal_label in args.metal_lines:
            # set f
            if args.fid_cont["n_f_" + metal_label] == 0:
                args.fid_cont["f_" + metal_label] = [0, -10.5]
            else:
                args.fid_cont["f_" + metal_label] = [
                    0,
                    fid_vals_metals["f_" + metal_label],
                ]
                if metal_label in ["SiIIa_SiIIb", "Lya_SiII", "Lya_SiIII"]:
                    args.fid_cont["n_s_" + metal_label] = 1

            if args.fid_cont["n_s_" + metal_label] == 0:
                args.fid_cont["s_" + metal_label] = [0, -10.5]
            else:
                args.fid_cont["s_" + metal_label] = [0, 4.5]

        for ii in range(4):
            if args.fid_cont["n_d_dla" + str(ii + 1)] == 0:
                args.fid_cont["HCD_damp" + str(ii + 1)] = [0, -10.5]
            else:
                args.fid_cont["HCD_damp" + str(ii + 1)] = [0, -2]

        like = set_like(
            data["P1Ds"],
            emulator,
            args,
            data_hires=data["extra_P1Ds"],
        )

        if iq > 0:
            for par in out["mle"][0]:
                try:
                    par2 = fitter.param_dict_rev[par]
                except:
                    continue
                for p in like.free_params:
                    if par2 == p.name:
                        p.value = out["mle"][0][par]
                        break

        print()
        f_space_len = 14
        s_space_len = 5
        for p in like.free_params:
            print(
                p.name,
                (f_space_len - len(p.name)) * " ",
                "\t",
                np.round(p.value, 3),
                (s_space_len - len(str(np.round(p.value, 3)))) * " ",
                "\t",
                np.round(p.min_value, 3),
                (s_space_len - len(str(np.round(p.min_value, 3)))) * " ",
                "\t",
                np.round(p.max_value, 3),
                (s_space_len - len(str(np.round(p.max_value, 3)))) * " ",
                "\t",
                p.Gauss_priors_width,
            )
        print()

        fitter = Fitter(
            like=like,
            rootdir=output_dir,
            nburnin=args.n_burn_in,
            nsteps=args.n_steps,
            parallel=args.parallel,
            explore=args.explore,
            fix_cosmology=args.fix_cosmo,
        )

        p0 = np.array(list(like.fid["fit_cube"].values()))

        zmask = np.array([like.data.z[iz]])
        fitter.run_minimizer(
            log_func_minimize=fitter.like.minus_log_prob,
            p0=p0,
            zmask=zmask,
            restart=True,
            nsamples=6,
        )

        print(fitter.mle_chi2)

        out["param_names"].append(list_props)
        out["mle"].append(fitter.mle)
        out["mle_cube"].append(fitter.mle_cube)
        out["chi2"].append(fitter.mle_chi2)
        out["ndeg"].append(len(like.data.k_kms[iz]) - len(list_props))

    np.save(folder + "grow_" + label_fit + ".npy", out)


def run_grow_model_atz(folder, zs, verbose=True):
    """Read"""
    select_props = {}
    for iz in range(len(zs)):
        # for iz in range(10, len(like.data.z)):
        if verbose:
            print(zs[iz])
        keep = True
        chi2_im = []
        prob_im = []

        fix_props = [
            "n_tau",
            "n_sigT",
            "n_f_Lya_SiIII",
            "n_s_Lya_SiIII",
            "n_d_dla1",
        ]

        basic_props = [
            None,
            "n_gamma",
            "n_kF",
            "n_f_Lya_SiII",
            "n_f_SiIIa_SiIII",
            "n_f_SiIIb_SiIII",
            "n_f_SiIIa_SiIIb",
            "n_d_dla2",
            "n_d_dla3",
            "n_d_dla4",
        ]
        it = 0
        while keep:
            label_fit = "iz_" + str(iz) + "_it_" + str(it)

            if it < 10:
                pass
            else:
                chi2_grow_model_atz(
                    folder,
                    args,
                    iz,
                    fix_props,
                    basic_props,
                    label_fit=label_fit,
                )

            res = np.load(
                folder + "grow_" + label_fit + ".npy", allow_pickle=True
            ).item()
            Dchi2 = np.zeros(len(res["chi2"]))

            for ii in range(1, len(res["chi2"])):
                Dchi2[ii] = res["chi2"][0] - res["chi2"][ii]
                if verbose:
                    print(res["param_names"][ii][-1], Dchi2[ii])
            ind = np.argmax(Dchi2)
            prob1 = chi2_scipy.sf(res["chi2"][0], res["ndeg"][0])
            prob2 = chi2_scipy.sf(res["chi2"][ind], res["ndeg"][ind])
            best_prop = res["param_names"][ind][-1]
            # print(res["mle_cube"][ind])
            if verbose:
                print(res["mle"][ind])

            if Dchi2[ind] == 0:
                break

            if it == 0:
                prob_im.append(prob1)
            else:
                chi2_im.append(Dchi2[ind])
                prob_im.append(prob2)

            if verbose:
                print()

            if prob2 > prob1:
                if verbose:
                    print(
                        best_prop, "\t", np.round(Dchi2[ind], 2), prob1, prob2
                    )
                fix_props.append(best_prop)
                basic_props.remove(best_prop)
                it += 1
            else:
                keep = False

            if verbose:
                print(fix_props)
                print()

        if verbose:
            print()
            print()
            print()
        labz = str(np.round(zs[iz], 2))
        select_props[labz] = {}
        select_props[labz]["z"] = zs[iz]
        select_props[labz]["name"] = np.array(fix_props)
        select_props[labz]["chi2"] = np.array(chi2_im)
        select_props[labz]["prob"] = np.array(prob_im)[: len(chi2_im)]

    return select_props
