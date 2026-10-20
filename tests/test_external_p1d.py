import pytest
"""The external observation API and native scalar API share algebra/order."""
import numpy as np
from cup1d.theory.theory import Theory
from cup1d.likelihood.external import apply_observation_model


class Contaminants:
    def get_contamination(self,zs,ks,flux,M,**kwargs):
        return {name:[np.full(len(k),v) for k in ks] for name,v in
                dict(cont_HCD=2.,cont_mul_metals=3.,IC_corr=5.,cont_add_metals=7.).items()}


class Systematics:
    def get_contamination(self,zs,ks,**kwargs):
        return [np.full(len(k),11.) for k in ks]


class Emulator:
    emulator_label="fixture"
    def emulate_p1d_Mpc(self,inputs,k):
        return k**2+1.


class FixtureTheory(Theory):
    def __init__(self):
        self.emulator=Emulator()
        self.model_cont=Contaminants()
        self.model_syst=Systematics()
        self.star_priors=None
        self.use_hull=False

    def get_emulator_calls(self,zs,**kwargs):
        return {"mF":np.full(len(zs),.7)},np.arange(len(zs))+70.,(0.,)*6


def test_native_scalar_external_order_and_return_contract():
    t=FixtureTheory()
    zs=np.array([2.,3.])
    ks=[np.array([.001,.002]),np.array([.001,.002,.003])]
    flux=np.full(2,.7)
    M=np.array([70.,71.])
    raw=[((k*m)**2+1)*m for k,m in zip(ks,M)]
    external,terms=apply_observation_model(t.model_cont,t.model_syst,zs,ks,raw,flux,M,{"R_coeff_0":0.})
    native=t.get_p1d_kms(zs,ks,like_params={"R_coeff_0":0.},return_blob=True,return_contaminants=True)
    for a,b in zip(external,native[0]):
        np.testing.assert_array_equal(a,b)
    assert native[1]==(0.,)*6
    assert native[-1][0]["IC_corr"][0]==5.
    assert terms[0]["p1d_tot_kms"]==native[-1][0]["p1d_tot_kms"]


def test_public_external_coefficient_route_returns_immutable_rows():
    """Supplied coefficients are projected without invoking an emulator."""
    from cup1d.likelihood.external import PredictionGroup, PredictionRequest, project_arinyo_request
    from lace.cosmo.cosmology import Cosmology

    class ProjectionCosmology(Cosmology):
        """LaCE-only cosmology fixture for the external coefficient boundary."""

        def validate_k(self, k_iMpc):
            if np.any(np.asarray(k_iMpc) > self.get_kmax_linP_Mpc()):
                raise ValueError("k outside the linear-power coverage")

    z = np.array([2.5])
    snapshot = ProjectionCosmology()
    request = PredictionRequest((PredictionGroup("fixture", (2.5,), ((.001, .002, .003),)),), "fixture")
    coefficients = {2.5: dict(bias=-.15, bias_eta=-.2, q1=.2, q2=.01,
                             kvav=.5, av=.3, bv=1.2, kp=8.)}
    result = project_arinyo_request(request, snapshot, coefficients,
                                    projection={"n_k_perp": 24, "k_perp_max_iMpc": 10.},
                                    kmax_iMpc=5.)
    assert result["fixture"][0].shape == (3,)
    assert np.all(np.isfinite(result["fixture"][0]))
    assert not result["fixture"][0].flags.writeable
    with np.testing.assert_raises(ValueError):
        result["fixture"][0].setflags(write=True)


@pytest.mark.parametrize("damping", ["sigmoid", "gaussian"])
@pytest.mark.parametrize("rule", ["k_perp", "k_mu"])
def test_kaiser_metal_moments_equal_direct_projection_of_contaminated_p3d(rule, damping):
    """Moments + SiKaiser reproduce the k_perp integral of the full contaminated P3D."""
    from cup1d.configuration.args import Args
    from cup1d.likelihood.external import (
        PredictionGroup, PredictionRequest, _quadrature, make_integrator,
        project_arinyo_request, project_metal_moments)
    from cup1d.models.contaminants.metals.si_kaiser import SiKaiser, lya_metal_damping, velocity_offset
    from forestflow.model.arinyo import ArinyoModel
    from forestflow.model.linear import LinearTheoryGrid
    from lace.cosmo.cosmology import Cosmology

    class ProjectionCosmology(Cosmology):
        def validate_k(self, k_iMpc):
            if np.any(np.asarray(k_iMpc) > self.get_kmax_linP_Mpc()):
                raise ValueError("k outside the linear-power coverage")

    z = 2.5
    snapshot = ProjectionCosmology()
    k_kms = np.geomspace(1e-3, 0.03, 40)
    request = PredictionRequest((PredictionGroup("fixture", (z,), (tuple(k_kms),)),), "fixture")
    arinyo = {z: dict(bias=-.15, bias_eta=-.2, q1=.2, q2=.01, kvav=.5, av=.3, bv=1.2, kp=8.)}
    projection = ({"rule": "k_mu", "k_perp_max_iMpc": 10.} if rule == "k_mu"
                  else {"n_k_perp": 120, "k_perp_max_iMpc": 10.})
    integrator = make_integrator(projection)
    lya = project_arinyo_request(request, snapshot, arinyo, kmax_iMpc=5., integrator=integrator)["fixture"][0]
    moments = project_metal_moments(request, snapshot, arinyo, kmax_iMpc=5., integrator=integrator)["fixture"][0]
    # Constant histories: ln|b| and ln(sigma) with exp otype, beta with const.
    coeffs = {"b_SiIII": [np.log(0.02)], "b_SiIIa": [np.log(0.015)], "b_SiIIb": [np.log(0.01)],
              "b_SiIIc": [np.log(0.012)], "beta_SiIII": [0.7], "beta_SiII": [1.1],
              "sig_SiIII": [np.log(80.)], "sig_SiII": [np.log(150.)],
              "f_SiIIxSiII": [0.0], "sig_SiIIxSiII": [-20.0]}  # inactive SiII x SiII factor
    prop_coeffs = {}
    for name, otype in Args._KAISER_OTYPES.items():
        prop_coeffs[name + "_ztype"] = "pivot"
        prop_coeffs[name + "_otype"] = otype
    metals = SiKaiser(coeffs=coeffs, prop_coeffs=prop_coeffs, flat_priors=Args._CONT_FLAT_PRIORS,
                      damping=damping)
    assert np.isclose(metals.coefficients(z)["SiIII"][0], -0.02)
    contaminated = lya + metals.get_contamination([z], [k_kms], metal_moments=[moments])[0]

    # Direct: build the contaminated P3D on the same quadrature and integrate it.
    M = float(snapshot.get_dkms_diMpc(np.array([z]))[0])
    model = ArinyoModel(fiducial_cosmology=snapshot)
    linear = LinearTheoryGrid(z=np.array([z]), cosmology=snapshot)
    k_par = (k_kms * M)[None, :]
    k3d, mu, integrate = _quadrature(integrator, k_par)
    k_par = k_par[..., None]
    mu2 = mu**2
    lin = model.linear.get_linP_Mpc(linear, np.array([z]), k3d)
    f = np.asarray(model.linear.get_fz(linear, np.array([z])))[:, None, None]
    pars = model.default_params | {k: np.asarray([v])[:, None, None] for k, v in arinyo[z].items()}
    dnl = model.nonlinear_correction(lin, k3d, mu, pars)
    leg = np.exp(-0.5 * (k3d / pars["kp"]) ** 2)  # pressure smoothing of one metal leg
    lya_k = pars["bias"] + pars["bias_eta"] * f * mu2
    par = metals.coefficients(z)
    kaiser = {l: b * (1 + beta * mu2) * leg for l, (b, beta, s) in par.items()}
    window = {l: lya_metal_damping(k_par / M, s, damping) for l, (b, beta, s) in par.items()}
    p3d = lin * dnl * lya_k**2
    lines = list(par)
    for l in lines:
        p3d = p3d + 2 * np.cos(k_par / M * velocity_offset("Lya", l)) * lin * np.sqrt(dnl) * lya_k * kaiser[l] * window[l]
    for i, a in enumerate(lines):  # metal x metal: no decorrelation
        p3d = p3d + lin * kaiser[a] ** 2
        for b in lines[i + 1:]:
            p3d = p3d + 2 * np.cos(k_par / M * velocity_offset(a, b)) * lin * kaiser[a] * kaiser[b]
    direct = integrate(p3d)[0] * M
    np.testing.assert_allclose(contaminated, direct, rtol=1e-10)
    assert np.max(np.abs(contaminated / lya - 1)) > 0.05  # metals are not negligible here


def test_kaiser_sii_x_sii_factor_acts_only_on_sii_x_sii_terms():
    """f_SiIIxSiII and sig_SiIIxSiII rescale and decorrelate only the SiII x SiII terms."""
    from cup1d.configuration.args import Args
    from cup1d.models.contaminants.metals.si_kaiser import SPECIES, SiKaiser

    k = np.geomspace(1e-3, 0.03, 7)
    moments = {name: np.linspace(1.0, 2.0, k.size) * (i + 1)
               for i, name in enumerate(("A0", "A1", "I0", "I1", "I2"))}
    coeffs = {"b_SiIII": [np.log(0.02)], "b_SiIIa": [np.log(0.015)], "b_SiIIb": [np.log(0.01)],
              "b_SiIIc": [np.log(0.012)], "beta_SiIII": [0.7], "beta_SiII": [1.1],
              "sig_SiIII": [np.log(80.)], "sig_SiII": [np.log(150.)]}
    prop_coeffs = {}
    for name, otype in Args._KAISER_OTYPES.items():
        prop_coeffs[name + "_ztype"] = "pivot"
        prop_coeffs[name + "_otype"] = otype

    def terms(f, sigma):
        extra = {"f_SiIIxSiII": [np.log(f)], "sig_SiIIxSiII": [np.log(sigma)]}
        model = SiKaiser(coeffs=coeffs | extra, prop_coeffs=prop_coeffs,
                         flat_priors=Args._CONT_FLAT_PRIORS)
        return model.p1d_terms(2.5, k, moments)

    base = terms(1.0, np.exp(-20.0))
    scaled = terms(3.0, 120.0)
    window = 3.0 * np.exp(-0.5 * (k * 120.0) ** 2)
    n_sii = 0
    for name, term in base.items():
        lines = name.split("_")
        if all(line in SPECIES and SPECIES[line] == "SiII" for line in lines):
            np.testing.assert_allclose(scaled[name], term * window, rtol=1e-12)
            n_sii += 1
        else:
            np.testing.assert_array_equal(scaled[name], term)
    assert n_sii == 6  # three SiII autos and three SiII x SiII crosses


def test_lya_metal_damping_forms():
    """The default is the Gaussian of Vega, the sigmoid that of CM2026; both are 1 at k = 0."""
    from cup1d.models.contaminants.metals.si_kaiser import lya_metal_damping

    k = np.array([0.0, 1e-3, 1e-2])
    np.testing.assert_allclose(lya_metal_damping(k, 150.0), np.exp(-0.5 * (150.0 * k) ** 2))
    np.testing.assert_allclose(lya_metal_damping(k, 150.0, "sigmoid"), 2 - 2 / (1 + np.exp(-150.0 * k)),
                               rtol=1e-15)
    assert lya_metal_damping(0.0, 150.0) == lya_metal_damping(0.0, 150.0, "sigmoid") == 1.0
    with pytest.raises(ValueError):
        lya_metal_damping(k, 150.0, "lorentzian")


def test_coefficient_families_match_exactly_not_by_substring():
    """sig_SiII and sig_SiIIxSiII keep their own priors, counts and overrides."""
    from copy import deepcopy

    from cup1d.configuration.args import Args
    from cup1d.models.contaminants.metals.si_kaiser import FAMILIES, SiKaiser

    priors = deepcopy(Args._CONT_FLAT_PRIORS)
    priors["sig_SiIIxSiII"] = [[-1, 1], [1.0, 6.0]]  # sig_SiII precedes it with [0, 7]
    prop_coeffs = {}
    for name, otype in Args._KAISER_OTYPES.items():
        prop_coeffs[name + "_ztype"] = "interp_lin"
        prop_coeffs[name + "_otype"] = otype
        prop_coeffs[name + "_znodes"] = [2.2, 3.0]
    fid_vals = {name: [0.0, 0.0] for name in FAMILIES}
    free = ["sig_SiII_0", "sig_SiII_1", "sig_SiIIxSiII_0", "sig_SiIIxSiII_1"]
    model = SiKaiser(free_param_names=free, fid_vals=fid_vals, prop_coeffs=prop_coeffs,
                     flat_priors=priors)
    assert model.n_pars["sig_SiII"] == model.n_pars["sig_SiIIxSiII"] == 2
    bounds = {name: (par["min_value"], par["max_value"]) for name, par in model.params.items()}
    assert bounds["sig_SiII_0"] == (0.0, 7.0)
    assert bounds["sig_SiIIxSiII_0"] == (1.0, 6.0)
    like_params = {"sig_SiII_0": 1.0, "sig_SiII_1": 2.0, "sig_SiIIxSiII_0": 3.0, "sig_SiIIxSiII_1": 4.0}
    np.testing.assert_array_equal(model.get_coeff("sig_SiII", like_params), [1.0, 2.0])
    np.testing.assert_array_equal(model.get_coeff("sig_SiIIxSiII", like_params), [3.0, 4.0])


@pytest.mark.parametrize("rule", ["k_perp", "k_mu"])
def test_fused_projection_matches_forestflow_and_interpolation_is_accurate(rule):
    """The one-pass Lya P1D equals ForestFlow's projection; n_k interpolation is accurate."""
    from cup1d.likelihood.external import (
        PredictionGroup, PredictionRequest, make_integrator, project_fused)
    from forestflow.model.arinyo import ArinyoModel
    from forestflow.model.linear import LinearTheoryGrid
    from lace.cosmo.cosmology import Cosmology

    class ProjectionCosmology(Cosmology):
        def validate_k(self, k_iMpc):
            if np.any(np.asarray(k_iMpc) > self.get_kmax_linP_Mpc()):
                raise ValueError("k outside the linear-power coverage")

    zs = (2.4, 3.6)  # unequal row lengths: ragged grids
    k_rows = (tuple(np.linspace(1e-3, 0.03, 700)), tuple(np.linspace(1.2e-3, 0.035, 900)))
    request = PredictionRequest((PredictionGroup("fixture", zs, k_rows),), "fixture")
    arinyo = {z: dict(bias=-.15, bias_eta=-.2, q1=.2, q2=.01, kvav=.5, av=.3, bv=1.2, kp=8.)
              for z in zs}
    snapshot = ProjectionCosmology()
    projection = {"rule": rule, "k_perp_max_iMpc": 10.}
    if rule == "k_perp":
        projection["n_k_perp"] = 200
    integrator = make_integrator(projection)
    p1d, moments = project_fused(request, snapshot, arinyo, kmax_iMpc=5., integrator=integrator,
                                 moments=True)
    model = ArinyoModel(fiducial_cosmology=snapshot)
    for z, k, row in zip(zs, k_rows, p1d["fixture"]):
        M = float(snapshot.get_dkms_diMpc(np.array([z]))[0])
        linear = LinearTheoryGrid(z=np.array([z]), cosmology=snapshot)
        pars = {name: np.asarray([value])[:, None, None] for name, value in arinyo[z].items()}
        reference = model.P1D_Mpc(linear, np.array([z]), (np.asarray(k) * M)[None, :], pars,
                                  integrator=integrator)[0] * M
        np.testing.assert_allclose(row, reference, rtol=1e-12)
    p1d_i, moments_i = project_fused(request, snapshot, arinyo, kmax_iMpc=5., integrator=integrator,
                                     moments=True, n_k=64)
    for row, row_i in zip(p1d["fixture"], p1d_i["fixture"]):
        assert row_i.shape == row.shape
        np.testing.assert_allclose(row_i, row, rtol=2e-4)
    for mom, mom_i in zip(moments["fixture"], moments_i["fixture"]):
        for name in mom:
            scale = np.max(np.abs(mom[name]))
            np.testing.assert_allclose(mom_i[name], mom[name], rtol=0, atol=2e-4 * scale)
