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
