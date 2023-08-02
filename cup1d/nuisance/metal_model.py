import numpy as np
import copy
from cup1d.likelihood import likelihood_parameter


class MetalModel(object):
    """ Model the contamination from Silicon Lya cross-correlations """

    def __init__(self,metal_label,lambda_rest=None,z_X=3.0,ln_X_coeff=None,
                free_param_names=None):
        """ Model the evolution of a metal contamination (SiII or SiIII).
        We use a power law around z_X=3. """

        # label identifying the metal line
        self.metal_label=metal_label
        if metal_label=="SiIII":
            self.lambda_rest=1206.50 # from McDonald et al. 2006)
            if lambda_rest:
                if lambda_rest != self.lambda_rest:
                    raise ValueError('inconsistent lambda_rest')
        else:
            if lambda_rest is None:
                raise ValueError('need to specify lambda_rest',metal_label)
            self.lambda_rest=lambda_rest
            
        # power law pivot point
        self.z_X=z_X

        # figure out parameters 
        if ln_X_coeff:
            if free_param_names is not None:
                raise ValueError('can not specify coeff and free_param_names')
            self.ln_X_coeff=ln_X_coeff
        else:
            if free_param_names:
                # figure out number of free params for this metal line
                param_tag='ln_'+metal_label
                print('metal tag',param_tag)
                n_X=len([p for p in free_param_names if param_tag in p])
            else:
                n_X=1
            # start with value from McDonald et al. (2006), and no z evolution
            self.ln_X_coeff=[0.0]*n_X
            self.ln_X_coeff[-1]=np.log(0.011)

        # store list of likelihood parameters (might be fixed or free)
        self.set_parameters()


    def get_Nparam(self):
        """Number of parameters in the model"""
        if len(self.ln_X_coeff) != len(self.X_params):
            raise ValueError("parameter size mismatch")
        return len(self.ln_X_coeff)


    def set_parameters(self):
        """Setup likelihood parameters for metal model"""

        self.X_params=[]
        Npar=len(self.ln_X_coeff)
        param_tag='ln_'+self.metal_label
        for i in range(Npar):
            name=param_tag+'_'+str(i)
            if i==0:
                # log of overall amplitude at z_X
                xmin=-20
                xmax=0
            else:
                xmin=-5
                xmax=5
            # note non-trivial order in coefficients
            value=self.ln_X_coeff[Npar-i-1]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            self.X_params.append(par)
        return


    def get_parameters(self):
        """Return likelihood parameters from the metal model"""
        return self.X_params


    def update_parameters(self,like_params):
        """Look for metal parameters in list of parameters"""

        Npar=self.get_Nparam()
        param_tag='ln_'+self.metal_label

        # loop over likelihood parameters
        for like_par in like_params:
            if param_tag in like_par.name:
                # make sure you find the parameter
                found=False
                # loop over parameters in metal model
                for ip in range(len(self.X_params)):
                    if self.X_params[ip].name == like_par.name:
                        if found:
                            raise ValueError('can not update parameter twice')
                        self.ln_X_coeff[Npar-ip-1]=like_par.value
                        found=True
                if not found:
                    raise ValueError('did not update parameter '+like_par.name)

        return


    def get_new_model(self,parameters=[]):
        """Return copy of model, updating values from list of parameters"""

        X = MetalModel(metal_label = self.metal_label, 
                        lambda_rest = self.lambda_rest, 
                        z_X = self.z_X,
                        ln_X_coeff = copy.deepcopy(self.ln_X_coeff))

        X.update_parameters(parameters)
        return X


    def get_amplitude(self,z):
        """ Amplitude of contamination at a given z """

        # Note that this represents "f" in McDonald et al. (2006)
        # It is later rescaled by <F> to compute "a" in eq. (15)

        xz=np.log((1+z)/(1+self.z_X))
        ln_poly=np.poly1d(self.ln_X_coeff)
        ln_out=ln_poly(xz)
        return np.exp(ln_out)


    def get_dv_kms(self):
        """ Velocity separation where the contamination is stronger """

        # these constants should be written elsewhere
        lambda_lya = 1215.67
        c_kms = 2.997e5

        # we should properly compute this (check McDonald et al. 2006)
        dv_kms = (lambda_lya-self.lambda_rest)/lambda_lya*c_kms

        return dv_kms


    def get_contamination(self,z,k_kms,mF):
        """ Multiplicative contamination at a given z and k (in s/km).
            The mean flux (mF) is used scale it (see McDonald et al. 2006)"""

        # Note that this represents "f" in McDonald et al. (2006)
        # It is later rescaled by <F> to compute "a" in eq. (15)
        f=self.get_amplitude(z)
        a=f/(1-mF)
        # v3 in McDonald et al. (2006)
        dv=self.get_dv_kms()

        return (1+a**2) + 2*a*np.cos(dv*k_kms)

