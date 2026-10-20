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


@pytest.mark.parametrize("rule", ["k_perp", "k_mu"])
def test_kaiser_metal_moments_equal_direct_projection_of_contaminated_p3d(rule):
    """Moments + SiKaiser reproduce the k_perp integral of the full contaminated P3D."""
    from cup1d.configuration.args import Args
    from cup1d.likelihood.external import (
        PredictionGroup, PredictionRequest, _quadrature, make_integrator,
        project_arinyo_request, project_metal_moments)
    from cup1d.models.contaminants.metals.si_kaiser import SiKaiser, velocity_offset
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
              "sig_SiIII": [np.log(80.)], "sig_SiII": [np.log(150.)]}
    prop_coeffs = {}
    for name, otype in Args._KAISER_OTYPES.items():
        prop_coeffs[name + "_ztype"] = "pivot"
        prop_coeffs[name + "_otype"] = otype
    metals = SiKaiser(coeffs=coeffs, prop_coeffs=prop_coeffs, flat_priors=Args._CONT_FLAT_PRIORS)
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
    window = {l: np.exp(-0.5 * (k_par / M * s) ** 2) for l, (b, beta, s) in par.items()}
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
