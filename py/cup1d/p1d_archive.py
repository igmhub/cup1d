import numpy as np
import matplotlib.pyplot as plt
import json


class ArchiveP1D(object):
    """Book-keeping of flux P1D measured in a suite of simulations."""

    def __init__(self,data):
        """Constructor base class to archive P1D from simulations"""

        print('Inside base class ArchiveP1D constructor')
        self.data=data
        print('Number of data points stored in archive =',len(self.data))


    def print_entry(self,entry,keys=['z','Delta2_p','n_p','alpha_p','f_p',
                                    'mF','sigT_Mpc','gamma','kF_Mpc']):
        """Print basic information about a particular entry in the archive"""

        if entry >= len(self.data):
            raise ValueError('{} entry does not exist in archive'.format(entry))

        data = self.data[entry]
        info='entry = {}'.format(entry)
        for key in keys:
            info += ', {} = {:.4f}'.format(key,data[key])
        print(info)
        return


    def plot_samples(self,param_1,param_2,tau_scalings=True,temp_scalings=True):
        """For parameter pair (param1,param2), plot each point in the archive.
            Use tau_scalings and temp_scalings to plot also post-processing."""

        emu_data=self.data
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
        emu_z=np.array([emu_data[i]['z'] for i in range(Nemu) if (
                                                mask_tau[i] & mask_temp[i])])

		# we will color code the points with redshift
        zmin=min(emu_z)
        zmax=max(emu_z)
        plt.scatter(emu_1,emu_2,c=emu_z,s=1,vmin=zmin, vmax=zmax)
        cbar=plt.colorbar()
        cbar.set_label("Redshift", labelpad=+1)
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.show()

        return
