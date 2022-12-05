import numpy as np
import matplotlib.pyplot as plt
from cup1d.data import p1d_data_Irsic2017
from cup1d.data import p1d_data_Walther2018
from cup1d.data import p1d_data_Chabanier2019

def plot_all_p1d(zmin=1.7,zmax=6.0,kmin=0.001,kmax=0.1,
        use_dimensionless=True,xlog=True,ylog=True):
    """Plot P1D from multiple measurements: 
        - Irsic et al. (2017) 
        - Walther et al. (2018)
        - Chabanier et al. (2019)
        If use_dimensionless, plot k*P(k)/pi."""

    # setup list containing all measurements
    keys = ['label','marker','data'] 
    datasets = [dict(zip(keys,['I17','o',p1d_data_Irsic2017.P1D_Irsic2017()])),
            dict(zip(keys,['W18','x',p1d_data_Walther2018.P1D_Walther2018()])),
            dict(zip(keys,['C19','*',p1d_data_Chabanier2019.P1D_Chabanier2019()]))]

    plot_multiple_p1d(datasets,zmin=zmin,zmax=zmax,kmin=kmin,kmax=kmax,
            use_dimensionless=use_dimensionless,xlog=xlog,ylog=ylog)


def plot_multiple_p1d(datasets,zmin=1.7,zmax=6.0,kmin=0.001,kmax=0.1,
        use_dimensionless=True,xlog=True,ylog=True):
    """Plot P1D from different datasets. Input datasets should be a list
        of dictionaries with ('label','marker','data'), with data being
        a p1d_data object. If use_dimensionless, plot k*P(k)/pi."""

    Ndata=len(datasets)
    for idata in range(Ndata):
        label=datasets[idata]['label']
        marker=datasets[idata]['marker']
        data=datasets[idata]['data']
        k_kms=data.k_kms
        kplot=(k_kms>kmin) & (k_kms<kmax)
        k_kms=k_kms[kplot]
        zs=data.z
        Nz=len(zs)
        for iz in range(Nz):
            z=zs[iz]
            if z < zmin: continue
            if z > zmax: continue
            Pk_kms=data.get_Pk_iz(iz)[kplot]
            err_Pk_kms=np.sqrt(np.diagonal(data.get_cov_iz(iz)))[kplot]    
            fact=k_kms/np.pi
            col = plt.cm.jet((z-zmin)/(zmax-zmin))
            plt.errorbar(k_kms,fact*Pk_kms,
                         color=col,marker=marker,ms=4.5,ls="none",
                         yerr=fact*err_Pk_kms,
                         label=label+' z = {}'.format(z))
    plt.legend()
    plt.yscale('log', nonpositive='clip')
    plt.xscale('log')
    plt.ylabel(r'$k P(k)/ \pi$')
