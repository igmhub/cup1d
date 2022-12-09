import numpy as np
import matplotlib.pyplot as plt
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from cup1d.nuisance import mean_flux_model
from cup1d.nuisance import thermal_model
from cup1d.nuisance import pressure_model
from cup1d.likelihood import CAMB_model


class Theory(object):
    """Translator between the likelihood object and the emulator. This object
    will map from a set of CAMB parameters directly to emulator calls, without
    going through our Delta^2_\star parametrisation """

    def __init__(self,zs,emulator=None,verbose=False,
                    mf_model_fid=None,T_model_fid=None,kF_model_fid=None,
                    cosmo_fid=None,free_param_names=None):
        """Setup object to compute predictions for the 1D power spectrum.
        Inputs:
            - zs: redshifts that will be evaluated
            - emulator: object to interpolate simulated p1d
            - verbose: print information, useful to debug
            - cosmo_fid: fiducial cosmology used for fixed parameters
        """

        self.verbose=verbose
        self.zs=zs
        self.emulator=emulator

        # specify pivot point used in emulator
        if self.emulator is None:
            print('using default values for emulator pivot point')
            self.emu_kp_Mpc=0.7
        else:
            self.emu_kp_Mpc=self.emulator.archive.kp_Mpc

        # specify pivot point used in compressed parameters
        self.z_star=3.0
        self.kp_kms=0.009

        # setup fiducial cosmology
        if not cosmo_fid:
            cosmo_fid=camb_cosmo.get_cosmology()

        # setup CAMB object for the fiducial cosmology
        self.cosmo_model_fid=CAMB_model.CAMBModel(zs=self.zs,
                    cosmo=cosmo_fid,z_star=self.z_star,kp_kms=self.kp_kms)

        # setup fiducial IGM models (from Gadget sims if not specified)
        if mf_model_fid:
            self.mf_model_fid = mf_model_fid
        else:
            self.mf_model_fid = mean_flux_model.MeanFluxModel(
                    free_param_names=free_param_names)
        if T_model_fid:
            self.T_model_fid = T_model_fid
        else:
            self.T_model_fid = thermal_model.ThermalModel()
        if kF_model_fid:
            self.kF_model_fid = kF_model_fid
        else:
            self.kF_model_fid = pressure_model.PressureModel()


    def fixed_background(self,like_params):
        """Check if any of the input likelihood parameters would change
            the background expansion of the fiducial cosmology"""

        # look for parameters that would change background
        for par in like_params:
            if par.name in ['ombh2','omch2','H0','mnu','cosmomc_theta']:
                return False

        return True


    def get_linP_Mpc_params_from_fiducial(self,like_params):
        """Recycle linP_Mpc_params from fiducial model, when only varying
            primordial power spectrum (As, ns, nrun)"""

        # make sure you are not changing the background expansion
        assert self.fixed_background(like_params)

        # get linP_Mpc_params from fiducial model (should be very fast)
        linP_Mpc_params=self.cosmo_model_fid.get_linP_Mpc_params(
                kp_Mpc=self.emu_kp_Mpc)
        if self.verbose: print('got linP_Mpc_params for fiducial model')

        # differences in primordial power (at CMB pivot point)
        ratio_As=1.0
        delta_ns=0.0
        delta_nrun=0.0
        for par in like_params:
            if par.name == 'As':
                fid_As = self.cosmo_model_fid.cosmo.InitPower.As
                ratio_As = par.value / fid_As
            if par.name == 'ns':
                fid_ns = self.cosmo_model_fid.cosmo.InitPower.ns
                delta_ns = par.value - fid_ns
            if par.name == 'nrun':
                fid_nrun = self.cosmo_model_fid.cosmo.InitPower.nrun
                delta_nrun = par.value - fid_nrun

        # pivot scale in primordial power
        ks_Mpc=self.cosmo_model_fid.cosmo.InitPower.pivot_scalar
        # logarithm of ratio of pivot points
        ln_kp_ks=np.log(self.emu_kp_Mpc/ks_Mpc)

        # compute scalings
        delta_alpha_p=delta_nrun
        delta_n_p=delta_ns + delta_nrun*ln_kp_ks
        ln_ratio_A_p=np.log(ratio_As)+(delta_ns+0.5*delta_nrun*ln_kp_ks)*ln_kp_ks

        # update values of linP_params at emulator pivot point, at each z
        for zlinP in linP_Mpc_params:
            zlinP['Delta2_p'] *= np.exp(ln_ratio_A_p)
            zlinP['n_p'] += delta_n_p
            zlinP['alpha_p'] += delta_alpha_p

        return linP_Mpc_params


    def get_emulator_calls(self,like_params=[],return_M_of_z=True,
            return_blob=False):
        """Compute models that will be emulated, one per redshift bin.
            - like_params identify likelihood parameters to use.
            - return_blob will return extra information about the call."""

        # setup IMG models using list of likelihood parameters
        igm_models=self.get_igm_models(like_params)
        mf_model=igm_models['mf_model']
        T_model=igm_models['T_model']
        kF_model=igm_models['kF_model']

        # compute linear power parameters at all redshifts, and H(z) / (1+z)
        if self.fixed_background(like_params):
            # use background and transfer functions from fiducial cosmology
            if self.verbose: print('recycle transfer function')
            linP_Mpc_params=self.get_linP_Mpc_params_from_fiducial(like_params)
            M_of_zs=self.cosmo_model_fid.get_M_of_zs()
            if return_blob:
                blob=self.get_blob_fixed_background(like_params)
        else:
            # setup a new CAMB_model from like_params
            if self.verbose: print('create new CAMB_model')
            camb_model=self.cosmo_model_fid.get_new_model(like_params)
            linP_Mpc_params=camb_model.get_linP_Mpc_params(
                    kp_Mpc=self.emu_kp_Mpc)
            M_of_zs=camb_model.get_M_of_zs()
            if return_blob:
                blob=self.get_blob(camb_model=camb_model)

        # loop over redshifts and store emulator calls
        emu_calls=[]
        Nz=len(self.zs)
        for iz,z in enumerate(self.zs):
            # emulator parameters for linear power, at this redshift (in Mpc)
            model=linP_Mpc_params[iz]
            # emulator parameters for nuisance models, at this redshift
            model['mF']=mf_model.get_mean_flux(z)
            model['gamma']=T_model.get_gamma(z)
            sigT_kms=T_model.get_sigT_kms(z)
            model['sigT_Mpc']=sigT_kms/M_of_zs[iz]
            kF_kms=kF_model.get_kF_kms(z)
            model['kF_Mpc']=kF_kms*M_of_zs[iz]
            if self.verbose: print(iz,z,'model',model)
            emu_calls.append(model)

        if return_M_of_z==True:
            if return_blob:
                return emu_calls,M_of_zs,blob
            else:
                return emu_calls,M_of_zs
        else:
            if return_blob:
                return emu_calls,blob
            else:
                return emu_calls


    def get_blobs_dtype(self):
        """Return the format of the extra information (blobs) returned
            by get_p1d_kms and used in emcee_sampler. """

        blobs_dtype = [('Delta2_star', float),('n_star', float),
                        ('alpha_star', float),('f_star', float),
                        ('g_star', float),('H0',float)]
        return blobs_dtype


    def get_blob(self,camb_model=None):
        """Return extra information (blob) for the emcee_sampler. """

        if camb_model is None:
            Nblob=len(self.get_blobs_dtype())
            if Nblob==1:
                return np.nan
            else:
                out=np.nan,*([np.nan]*(Nblob-1))
                return out
        else:
            # compute linear power parameters for input cosmology
            params=self.cosmo_model_fid.get_linP_params()
            return params['Delta2_star'],params['n_star'], \
                    params['alpha_star'],params['f_star'], \
                    params['g_star'],camb_model.cosmo.H0


    def get_blob_fixed_background(self,like_params):
        """Fast computation of blob when running with fixed background"""

        # make sure you are not changing the background expansion
        assert self.fixed_background(like_params)

        # differences in primordial power (at CMB pivot point)
        ratio_As=1.0
        delta_ns=0.0
        delta_nrun=0.0
        for par in like_params:
            if par.name == 'As':
                fid_As = self.cosmo_model_fid.cosmo.InitPower.As
                ratio_As = par.value / fid_As
            if par.name == 'ns':
                fid_ns = self.cosmo_model_fid.cosmo.InitPower.ns
                delta_ns = par.value - fid_ns
            if par.name == 'nrun':
                fid_nrun = self.cosmo_model_fid.cosmo.InitPower.nrun
                delta_nrun = par.value - fid_nrun

        # pivot scale of primordial power
        ks_Mpc=self.cosmo_model_fid.cosmo.InitPower.pivot_scalar

        # likelihood pivot point, in velocity units
        dkms_dMpc=self.cosmo_model_fid.dkms_dMpc(self.z_star)
        kp_Mpc=self.kp_kms*dkms_dMpc

        # logarithm of ratio of pivot points
        ln_kp_ks=np.log(kp_Mpc/ks_Mpc)

        # get blob for fiducial cosmo
        fid_blob=self.get_blob(self.cosmo_model_fid)

        # rescale blobs
        delta_alpha_star=delta_nrun
        delta_n_star=delta_ns+delta_nrun*ln_kp_ks
        ln_ratio_A_star=np.log(ratio_As)+(delta_ns+0.5*delta_nrun*ln_kp_ks)*ln_kp_ks

        alpha_star=fid_blob[2]+delta_alpha_star
        n_star=fid_blob[1]+delta_n_star
        Delta2_star=fid_blob[0]*np.exp(ln_ratio_A_star)

        return (Delta2_star, n_star, alpha_star) + fid_blob[3:]


    def get_p1d_kms(self,k_kms,like_params=[],return_covar=False,
                    return_blob=False):
        """Emulate P1D in velocity units, for all redshift bins,
            as a function of input likelihood parameters.
            It might also return a covariance from the emulator,
            or a blob with extra information for the emcee_sampler."""

        if self.emulator is None:
            raise ValueError('no emulator provided')

        # figure out emulator calls, one per redshift
        if return_blob:
            emu_calls,M_of_z,blob=self.get_emulator_calls(
                    like_params=like_params,
                    return_M_of_z=True,return_blob=True)
        else:
            emu_calls,M_of_z=self.get_emulator_calls(
                    like_params=like_params,
                    return_M_of_z=True,return_blob=False)

        # loop over redshifts and compute P1D
        p1d_kms=[]
        if return_covar:
            covars=[]
        Nz=len(self.zs)
        for iz,z in enumerate(self.zs):
            # will call emulator for this model
            model=emu_calls[iz]
            # emulate p1d
            k_Mpc = k_kms * M_of_z[iz]
            if return_covar:
                p1d_Mpc, cov_Mpc = self.emulator.emulate_p1d_Mpc(model,k_Mpc,
                                                        return_covar=True,
                                                        z=z)
            else:
                p1d_Mpc = self.emulator.emulate_p1d_Mpc(model,k_Mpc,
                                                        return_covar=False,
                                                        z=z)
            if p1d_Mpc is None:
                if self.verbose: print('emulator did not provide P1D')
                p1d_kms.append(None)
                if return_covar:
                    covars.append(None)
            else:
                p1d_kms.append(p1d_Mpc * M_of_z[iz])
                if return_covar:
                    if cov_Mpc is None:
                        covars.append(None)
                    else:
                        covars.append(cov_Mpc * M_of_z[iz]**2)

        # decide what to return, and return it
        if return_covar:
            if return_blob:
                return p1d_kms,covars,blob
            else:
                return p1d_kms,covars
        else:
            if return_blob:
                return p1d_kms,blob
            else:
                return p1d_kms


    def get_parameters(self):
        """Return parameters in models, even if not free parameters"""

        # get parameters from CAMB model
        params=self.cosmo_model_fid.get_likelihood_parameters()

        # get parameters from nuisance models
        for par in self.mf_model_fid.get_parameters():
            params.append(par)
        for par in self.T_model_fid.get_sigT_kms_parameters():
            params.append(par)
        for par in self.T_model_fid.get_gamma_parameters():
            params.append(par)
        for par in self.kF_model_fid.get_parameters():
            params.append(par)

        if self.verbose:
            print('got parameters')
            for par in params:
                print(par.info_str())

        return params


    def get_igm_models(self,like_params=[]):
        """Setup IGM models from input list of likelihood parameters"""

        mf_model = self.mf_model_fid.get_new_model(like_params)
        T_model = self.T_model_fid.get_new_model(like_params)
        kF_model = self.kF_model_fid.get_new_model(like_params)

        models={'mf_model':mf_model,'T_model':T_model,'kF_model':kF_model}

        return models


    def plot_p1d(self,k_kms,like_params=[],plot_every_iz=1):
        """Emulate and plot P1D in velocity units, for all redshift bins,
            as a function of input likelihood parameters"""

        # ask emulator prediction for P1D in each bin
        emu_p1d=self.get_p1d_kms(k_kms,like_params)

        # plot only few redshifts for clarity
        Nz=len(self.zs)
        for iz in range(0,Nz,plot_every_iz):
            # acess data for this redshift
            z=self.zs[iz]
            p1d=emu_p1d[iz]
            # plot everything
            col = plt.cm.jet(iz/(Nz-1))
            plt.plot(k_kms,p1d*k_kms/np.pi,color=col,label='z=%.1f'%z)
        plt.yscale('log')
        plt.legend()
        plt.xlabel('k [s/km]')
        plt.ylabel(r'$k_\parallel \, P_{\rm 1D}(z,k_\parallel) / \pi$')
        plt.ylim(0.005,0.6)
        plt.show()

        return
