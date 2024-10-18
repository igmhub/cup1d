# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---



# ### Investigate metals

from cup1d.nuisance import metal_model

# +
z = np.linspace(2, 5)

for ii in range(-5, 5):
    fid_SiIII=[ii, -5]
    SiIII_model = metal_model.MetalModel(
        metal_label="SiIII",
        free_param_names=free_parameters,
        fid_value=fid_SiIII,
    )
    y = SiIII_model.get_amplitude(z)
    plt.plot(z, y, label=str(ii))
# -



like.fid["igm"]

# +
z = np.linspace(2, 5, 7)

k_kms = np.linspace(np.min(data["P1Ds"].k_kms[0]), np.max(data["P1Ds"].k_kms[0]), 100)

for jj in range(2):
    if(jj == 0):
        fid_SiIII=[0, -5]
    else:
        fid_SiIII=[-1, -5]
    
    SiIII_model = metal_model.MetalModel(
        metal_label="SiIII",
        free_param_names=free_parameters,
        fid_value=fid_SiIII,
    )
    for ii in range(5):
        col = "C"+str(ii)
        mF = like.theory.model_igm.F_model.get_mean_flux(z[ii])
        y = SiIII_model.get_contamination(
            z=z[ii],
            k_kms=k_kms,
            mF=mF,
        )
        # if ii == -5:
            # y0 = y.copy()
        if(jj == 0):
            plt.plot(k_kms, y, col, label=str(z[ii]))
        else:
            plt.plot(k_kms, y, col+'--', label=str(z[ii]))
# plt.legend()
