import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.optimize import minimize

from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP

from cup1d.likelihood import cosmologies
from cup1d.likelihood import lya_theory
from cup1d.nuisance import mean_flux_model
from cup1d.nuisance import thermal_model
from cup1d.nuisance import pressure_model


class Likelihood(object):
    """Likelihood class, holds data, theory, and knows about parameters"""

    def __init__(self,data,theory=None,emulator=None,
                    cosmo_fid_label='default',
                    free_param_names=None,
                    free_param_limits=None,
                    verbose=False,
                    prior_Gauss_rms=0.2,
                    kmin_kms=None,
                    emu_cov_factor=1,
                    extra_p1d_data=None,
                    min_log_like=-1e100,
                    fid_igm_fname=None):
        """Setup likelihood from theory and data. Options:
            - data (required) is the data to model
            - theory (optional) if not provided, will setup using emulator and
              list of free parameters
            - emulator (optional) only needed if theory not provided
            - cosmo_fid_label (optional) to specify fiducial cosmology
                        default: use default Planck-like cosmology
                        truth: read true cosmology used in simulation
                        look at cosmologies.py for more options
            - free_param_names is a list of param names, in any order
            - free_param_limits list of tuples, same order than free_param_names
            - if prior_Gauss_rms is None it will use uniform priors
            - ignore k-bins with k > kmin_kms
            - emu_cov_factor adjusts the contribution from emulator covariance
            set between 0 and 1.
            - extra_p1d_data: extra P1D data, e.g., from HIRES
            - min_log_like: use this instead of - infinity
            - fid_igm_fname: specify file with fiducial IGM models """

        self.verbose=verbose
        self.prior_Gauss_rms=prior_Gauss_rms
        self.emu_cov_factor=emu_cov_factor
        self.cosmo_fid_label=cosmo_fid_label
        self.min_log_like=min_log_like
        self.fid_igm_fname=fid_igm_fname
        self.data=data
        # (optionally) get rid of low-k data points
        self.data._cull_data(kmin_kms)

        if theory:
            assert cosmo_fid_label=='default', "wrong settings"
            self.theory=theory
        else:
            if cosmo_fid_label=='truth':
                # use true cosmology as fiducial (mostly for debugging)
                cosmo_fid=self.get_sim_cosmo()
                print('use true cosmo')
                camb_cosmo.print_info(cosmo_fid)
            elif cosmo_fid_label=='default':
                cosmo_fid=None
            else:
                cosmo_fid=cosmologies.get_cosmology_from_label(cosmo_fid_label)
                print('specified fiducial cosmology')
                camb_cosmo.print_info(cosmo_fid)

            # figure out which IGM fiducial model to use
            if fid_igm_fname:
                # for now this is only used in Nyx sims
                print('create Nyx IGM models')
                F_fid=mean_flux_model.MeanFluxModel(fid_fname=fid_igm_fname,
                        free_param_names=free_param_names)
                T_fid=thermal_model.ThermalModel(fid_fname=fid_igm_fname,
                        free_param_names=free_param_names)
                P_fid=pressure_model.PressureModel(fid_fname=fid_igm_fname,
                        free_param_names=free_param_names)
            else:
                # if not specified, use Gadget models
                F_fid=None
                T_fid=None
                P_fid=None

            # Set up a theory object
            self.theory=lya_theory.Theory(zs=self.data.z,
                    emulator=emulator,verbose=verbose,
                    cosmo_fid=cosmo_fid,F_model_fid=F_fid,
                    T_model_fid=T_fid,P_model_fid=P_fid,
                    free_param_names=free_param_names)

        # setup parameters
        self.set_free_parameters(free_param_names,free_param_limits)
        if verbose: print(len(self.free_params),'free parameters')

        # extra P1D likelihood from, e.g., HIRES
        if extra_p1d_data:
            include_metals=[]
            for metal in self.theory.metal_models:
                include_metals.append(metal.metal_label)
            # new theory, since we might need different zs
            extra_theory=lya_theory.Theory(zs=extra_p1d_data.z,
                    emulator=self.theory.emulator,
                    cosmo_fid=self.theory.cosmo_model_fid.cosmo,
                    F_model_fid=self.theory.F_model_fid,
                    T_model_fid=self.theory.T_model_fid,
                    P_model_fid=self.theory.P_model_fid,
                    include_metals=include_metals,
                    verbose=verbose)
            self.extra_p1d_like=Likelihood(data=extra_p1d_data,
                    theory=extra_theory,emulator=None,
                    free_param_names=free_param_names,
                    free_param_limits=free_param_limits,
                    verbose=verbose,
                    prior_Gauss_rms=prior_Gauss_rms,
                    kmin_kms=kmin_kms,
                    emu_cov_factor=emu_cov_factor,
                    extra_p1d_data=None)
        else:
            self.extra_p1d_like=None

        # sometimes we want to know the true theory (when working with mocks)
        self.set_truth()

        return


    def set_free_parameters(self,free_param_names,free_param_limits):
        """Setup likelihood parameters that we want to vary"""

        # setup list of likelihood free parameters
        self.free_params=[]

        if free_param_limits is not None:
            assert len(free_param_limits)==len(free_param_names), "wrong number of parameter limits"

        # get all parameters in theory, free or not
        params = self.theory.get_parameters()

        ## select free parameters, make sure ordering
        ## in self.free_params is same as in free_param_names
        for par_name in free_param_names:
            for par in params:
                if par.name == par_name:
                    if free_param_limits is not None:
                        ## Set min and max of each parameter if
                        ## a list is given. otherwise leave as default
                        par.min_value=free_param_limits[free_param_names.index(par.name)][0]
                        par.max_value=free_param_limits[free_param_names.index(par.name)][1]
                    self.free_params.append(par)

        Nfree=len(self.free_params)
        Nin=len(free_param_names)

        assert (Nfree==Nin), 'could not setup free parameters'

        if self.verbose:
            print('likelihood setup with {} free parameters'.format(Nfree))

        return


    def parameters_from_sampling_point(self,values):
        """Translate input array of values (in cube) to likelihood parameters"""
        
        if values is None:
            return []

        assert len(values)==len(self.free_params),'size mismatch'
        Npar=len(values)
        like_params=[]
        for ip in range(Npar):
            par = self.free_params[ip].get_new_parameter(values[ip])
            like_params.append(par)

        return like_params


    def cosmology_params_from_sampling_point(self,values):
        """ For a given point in sampling space, return a list of 
        cosmology params """

        like_params=self.parameters_from_sampling_point(values)

        ## Dictionary of cosmology parameters
        cosmo_dict={}

        for like_param in like_params:
            if like_param.name=="ombh2":
                cosmo_dict["ombh2"]=like_param.value
            elif like_param.name=="omch2":
                cosmo_dict["omch2"]=like_param.value
            elif like_param.name=="cosmomc_theta":
                cosmo_dict["cosmomc_theta"]=like_param.value
            elif like_param.name=="As":
                cosmo_dict["As"]=like_param.value
            elif like_param.name=="ns":
                cosmo_dict["ns"]=like_param.value
            elif like_param.name=="mnu":
                cosmo_dict["mnu"]=like_param.value
            elif like_param.name=="nrun":
                cosmo_dict["nrun"]=like_param.value

        assert len(cosmo_dict)>0, "No cosmology parameters found in sampling space"

        return cosmo_dict


    def get_sim_cosmo(self):
        """ Check that we are running on mock data, and return sim cosmo"""

        # different type of data will check for different sim cosmo
        if hasattr(self.data,"theory"):
            # using a mock_data P1D (computed from theory)
            return self.data.theory.cosmo_model_fid.cosmo
        elif hasattr(self.data,"sim_cosmo"):
            # using new data_gadget module
            return self.data.sim_cosmo
        else:
            # when working with real data can not return truth
            return None


    def set_truth(self,z_star=3.0,kp_kms=0.009):
        """ Store true cosmology from the simulation used to make mock data"""

        # access true cosmology used in mock data
        sim_cosmo=self.get_sim_cosmo()
        if sim_cosmo is None:
            print("will not store truth, working with real data")
            self.truth=None
            return
        camb_results_sim=camb_cosmo.get_camb_results(sim_cosmo,zs=[z_star])

        # store relevant parameters
        self.truth={}
        self.truth["ombh2"]=sim_cosmo.ombh2
        self.truth["omch2"]=sim_cosmo.omch2
        self.truth["As"]=sim_cosmo.InitPower.As
        self.truth["ns"]=sim_cosmo.InitPower.ns
        self.truth["nrun"]=sim_cosmo.InitPower.nrun
        self.truth["H0"]=sim_cosmo.H0
        self.truth["mnu"]=camb_cosmo.get_mnu(sim_cosmo)
        self.truth["cosmomc_theta"]=camb_results_sim.cosmomc_theta()

        # Store truth for compressed parameters
        linP_sim=fit_linP.parameterize_cosmology_kms(cosmo=sim_cosmo,
                        camb_results=camb_results_sim,
                        z_star=z_star,
                        kp_kms=kp_kms)
        self.truth["Delta2_star"]=linP_sim["Delta2_star"]
        self.truth["n_star"]=linP_sim["n_star"]
        self.truth["alpha_star"]=linP_sim["alpha_star"]
        self.truth["f_star"]=linP_sim["f_star"]
        self.truth["g_star"]=linP_sim["g_star"]


    def get_p1d_kms(self,k_kms=None,values=None,return_covar=False,
                    return_blob=False):
        """Compute theoretical prediction for 1D P(k)"""

        if k_kms is None:
            k_kms=self.data.k_kms

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params= self.parameters_from_sampling_point(values)
        else:
            like_params=[]

        return self.theory.get_p1d_kms(k_kms,like_params=like_params,
                                            return_covar=return_covar,
                                            return_blob=return_blob)


    def get_chi2(self,values=None):
        """Compute chi2 using data and theory, without adding
            emulator covariance"""

        log_like=self.get_log_like(values,ignore_log_det_cov=True)
        if log_like is None:
            return None
        else:
            return -2.0*log_like


    def get_log_like(self,values=None,ignore_log_det_cov=True,
            return_blob=False):
        """Compute log(likelihood), including determinant of covariance
            unless you are setting ignore_log_det_cov=True."""

        # get measured bins from data
        k_kms=self.data.k_kms
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        if return_blob:
            emu_p1d,emu_covar,blob=self.get_p1d_kms(k_kms,values,
                            return_covar=True,return_blob=True)
        else:
            emu_p1d,emu_covar=self.get_p1d_kms(k_kms,values,
                            return_covar=True,return_blob=False)

        if self.verbose: print('got P1D from emulator')

        # compute log like contribution from each redshift bin
        log_like=0

        for iz in range(Nz):
            # acess data for this redshift
            z=zs[iz]
            # make sure that theory is valid
            if emu_p1d[iz] is None:
                if self.verbose: print(z,'theory did not emulate p1d')
                return None
            if self.verbose: print('compute chi2 for z={}'.format(z))
            # get data
            p1d=self.data.get_Pk_iz(iz)
            data_cov=self.data.get_cov_iz(iz)
            # add covariance from emulator
            cov = data_cov + self.emu_cov_factor*emu_covar[iz]

            # compute chi2 for this redshift bin
            icov = np.linalg.inv(cov)
            diff = (p1d-emu_p1d[iz])
            chi2_z = np.dot(np.dot(icov,diff),diff)
            # check whether to add determinant of covariance as well
            if ignore_log_det_cov:
                log_like_z = -0.5*chi2_z
            else:
                (_, log_det_cov) = np.linalg.slogdet(cov)
                log_like_z = -0.5*(chi2_z + log_det_cov)
            log_like += log_like_z
            if self.verbose: print('added {} to log_like'.format(log_like_z))


        if return_blob:
            return log_like,blob
        else:
            return log_like


    def regulate_log_like(self,log_like):
        """Make sure that log_like is not NaN, nor tiny"""

        if (log_like is None) or math.isnan(log_like):
            return self.min_log_like

        return max(self.min_log_like,log_like)


    def compute_log_prob(self,values,return_blob=False):
        """Compute log likelihood plus log priors for input values
            - if return_blob==True, it will return also extra information"""

        # Always force parameter to be within range (for now)
        if (max(values) > 1.0) or (min(values) < 0.0):
            if return_blob:
                dummy_blob=self.theory.get_blob()
                return self.min_log_like, dummy_blob
            else:
                return self.min_log_like

        # compute log_prior
        log_prior=self.get_log_prior(values)

        # compute log_like (option to ignore emulator covariance)
        if return_blob:
            log_like,blob=self.get_log_like(values,ignore_log_det_cov=False,
                                            return_blob=True)
        else:
            log_like=self.get_log_like(values,ignore_log_det_cov=False,
                                            return_blob=False)

        # if required, add extra P1D likelihood from, e.g., HIRES
        if self.extra_p1d_like:
            extra_log_like=self.extra_p1d_like.get_log_like(values,
                        ignore_log_det_cov=False,return_blob=False)
            log_like += extra_log_like

        # regulate log-like (not NaN, not tiny)
        log_like=self.regulate_log_like(log_like)

        if return_blob:
            return log_like + log_prior, blob
        else:
            return log_like + log_prior


    def log_prob(self,values):
        """Return log likelihood plus log priors"""

        return self.compute_log_prob(values,return_blob=False)


    def log_prob_and_blobs(self,values):
        """Function used by emcee to get both log_prob and extra information"""

        lnprob,blob=self.compute_log_prob(values,return_blob=True)
        # unpack tuple
        out=lnprob,*blob
        return out


    def get_log_prior(self,values):
        """Compute logarithm of prior"""

        assert len(values)==len(self.free_params),'size mismatch'

        # Always force parameter to be within range (for now)
        if max(values) > 1.0:
            return self.min_log_like
        if min(values) < 0.0:
            return self.min_log_like

        if self.prior_Gauss_rms is None:
            return 0
        else:
            rms=self.prior_Gauss_rms
            fid_values=[p.value_in_cube() for p in self.free_params]
            log_prior=-np.sum((np.array(fid_values)-values)**2/(2*rms**2))
            return log_prior


    def minus_log_prob(self,values):
        """Return minus log_prob (needed to maximise posterior)"""

        return -1.0*self.log_prob(values)


    def maximise_posterior(self,initial_values=None,method='nelder-mead',tol=1e-4):
        """Run scipy minimizer to find maximum of posterior"""

        if not initial_values:
            initial_values=np.ones(len(self.free_params))*0.5

        return minimize(self.minus_log_prob, x0=initial_values,method=method,tol=tol)


    def plot_p1d(self,values=None,plot_every_iz=1,residuals=False,
                 plot_fname=None):
        """Plot P1D in theory vs data. If plot_every_iz >1,
            plot only few redshift bins"""

        # get measured bins from data
        k_kms=self.data.k_kms
        k_emu_kms=np.logspace(np.log10(min(k_kms)),np.log10(max(k_kms)),500)
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d, emu_cov = self.get_p1d_kms(k_emu_kms,values,return_covar=True)
        like_params=self.parameters_from_sampling_point(values)
        emu_calls=self.theory.get_emulator_calls(like_params)
        if self.verbose: print('got P1D from emulator')

        # plot only few redshifts for clarity
        for iz in range(0,Nz,plot_every_iz):
            # acess data for this redshift
            z=zs[iz]
            p1d_data=self.data.get_Pk_iz(iz)
            p1d_cov=self.data.get_cov_iz(iz)
            p1d_err=np.sqrt(np.diag(p1d_cov))
            p1d_theory=emu_p1d[iz]
            cov_theory=emu_cov[iz]
            err_theory=np.sqrt(np.diag(cov_theory))
            
            if p1d_theory is None:
                if self.verbose: print(z,'emulator did not provide P1D')
                continue
            # plot everything
            col = plt.cm.jet(iz/(Nz-1))

            if residuals:
                # interpolate theory to data kp values
                model=np.interp(k_kms,k_emu_kms,p1d_theory)
                # shift data in y axis for clarity
                yshift=iz/(Nz-1)
                plt.errorbar(k_kms,p1d_data/model+yshift,color=col,
                        yerr=p1d_err/model,
                        fmt="o",ms="4",label="z=%.1f" % z)
                plt.plot(k_emu_kms,p1d_theory/p1d_theory+yshift,
                        color=col,linestyle="dashed")
                plt.fill_between(k_emu_kms,
                        (p1d_theory+err_theory)/p1d_theory+yshift,
                        (p1d_theory-err_theory)/p1d_theory+yshift,
                        alpha=0.35,color=col)
            else:
                plt.errorbar(k_kms,p1d_data*k_kms/np.pi,color=col,
                        yerr=p1d_err*k_kms/np.pi,
                        fmt="o",ms="4",label="z=%.1f" % z)
                plt.plot(k_emu_kms,(p1d_theory*k_emu_kms)/np.pi,
                        color=col,linestyle="dashed")
                plt.fill_between(k_emu_kms,(p1d_theory+err_theory)*k_emu_kms/np.pi,
                        (p1d_theory-err_theory)*k_emu_kms/np.pi,
                        alpha=0.35,color=col)

        if residuals:
            plt.ylabel(r'$P_{\rm 1D}(z,k_\parallel)$ residuals')
            plt.ylim(0.9,2.1)
        else:
            plt.yscale('log')
            plt.ylabel(r'$k_\parallel \, P_{\rm 1D}(z,k_\parallel) / \pi$')
            plt.ylim(0.005,0.6)

        plt.plot(-10,-10,linestyle="-",label="Data",color="k")
        plt.plot(-10,-10,linestyle=":",label="Fit",color="k")
        plt.legend()
        plt.xlabel(r'$k_\parallel$ [s/km]')
        plt.xlim(min(k_kms)-0.001,max(k_kms)+0.001)
        plt.tight_layout()
        if plot_fname:
            plt.savefig(plot_fname)
        plt.show()

        return


    def overplot_emulator_calls(self,param_1,param_2,values=None,
                                tau_scalings=True,temp_scalings=True):
        """For parameter pair (param1,param2), overplot emulator calls
            with values stored in archive, color coded by redshift"""

        # mask post-process scalings (optional)
        emu_data=self.theory.emulator.archive.data
        Nemu=len(emu_data)
        if not tau_scalings:
            mask_tau=[x['scale_tau']==1.0 for x in emu_data]
        else:
            mask_tau=[True]*Nemu
        if not temp_scalings:
            mask_temp=[(x['scale_T0']==1.0) 
                        & (x['scale_gamma']==1.0) for x in emu_data]
        else:
            mask_temp=[True]*Nemu

        # figure out values of param_1,param_2 in archive
        emu_1=np.array([emu_data[i][param_1] for i in range(Nemu) if (
                                                  mask_tau[i] & mask_temp[i])])
        emu_2=np.array([emu_data[i][param_2] for i in range(Nemu) if (
                                                  mask_tau[i] & mask_temp[i])])

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params= self.parameters_from_sampling_point(values)
        else:
            like_params=[]
        emu_calls=self.theory.get_emulator_calls(like_params=like_params)
        # figure out values of param_1,param_2 called
        call_1=[emu_call[param_1] for emu_call in emu_calls]
        call_2=[emu_call[param_2] for emu_call in emu_calls]

        # overplot
        zs=self.data.z
        emu_z=np.array([emu_data[i]['z'] for i in range(Nemu) if (
                                                  mask_tau[i] & mask_temp[i])])
        zmin=min(min(emu_z),min(zs))
        zmax=max(max(emu_z),max(zs))
        plt.scatter(emu_1,emu_2,c=emu_z,s=1,vmin=zmin, vmax=zmax)
        plt.scatter(call_1,call_2,c=zs,s=50,vmin=zmin, vmax=zmax)
        cbar=plt.colorbar()
        cbar.set_label("Redshift", labelpad=+1)
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.show()

        return
