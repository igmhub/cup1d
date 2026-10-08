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
    from lya_interface.cosmology import CobayaCosmologySnapshot

    z = np.array([2.5])
    k = np.geomspace(.001, 100., 512)
    snapshot = CobayaCosmologySnapshot(z, k, np.tile(1e3*k**-2.5, (1, 1)), [250.], [.95])
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
