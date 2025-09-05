# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# # Check confidence levels

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from cup1d.utils.fit_ellipse import fit_ellipse
from scipy.stats import chi2 as chi2_scipy


# +
cont = np.array([0, 1, 2])
prob_levels = np.zeros(len(cont))
chi2_levels = np.zeros(len(cont))

prob_levels_2d = np.zeros(len(cont))

for ii in range(len(cont)):
    prob = chi2_scipy.cdf(cont[ii]**2, 1)
    chi2 = chi2_scipy.ppf(prob, 2)
    print(cont[ii], cont[ii]**2, chi2, prob)
    prob_levels[ii] = prob
    chi2_levels[ii] = chi2
    prob_levels_2d[ii] = chi2_scipy.cdf(cont[ii]**2, 2)

print(prob_levels)
print(prob_levels_2d)
print(chi2_levels)
# -

# ## 1D

# Data
nn = 1000000
x = np.linspace(-3, 3, nn)
# Multivariate Normal
mu_x = 0
sigma_x = 1
pre1D = np.sqrt(2 * np.pi)
rv1D = multivariate_normal(mu_x, sigma_x)
# Probability Density
pd_1D = rv1D.pdf(x)

levels = np.percentile(pd_1D, prob_levels * 100)
for ii in range(1, 3):
    print(ii, np.sum(np.abs(pd_1D < levels[ii]))/pd_1D.shape[0])

# ### 2D

# +
# Data
nn = 1000
x = np.linspace(-3, 3, nn)
y = np.linspace(-3, 3, nn)
X, Y = np.meshgrid(x,y)

# Multivariate Normal
mu_x = 0
sigma_x = 1
mu_y = 0
sigma_y = 1
pre = np.sqrt((2 * np.pi)**2)

rv = multivariate_normal([mu_x, mu_y], [[sigma_x, 0], [0, sigma_y]])

# Probability Density
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
pd = rv.pdf(pos) * pre

# +
levels = np.percentile(pd, prob_levels_2d * 100)
print(levels)                       

CS = plt.contour(
    X,
    Y,
    pd,
    levels,
    colors=["C1", "C0"],
)

for jj in range(2,3):
    p = CS.collections[jj].get_paths()
    x = []
    y = []
    for ii in range(len(p)):
        v = p[ii].vertices
        x.append(v[:,0])
        y.append(v[:,1])
    x = np.concatenate(x)
    y = np.concatenate(y)
    
    xfit, yfit, fit_params = fit_ellipse(x, y)
    plt.plot(xfit, yfit, "C"+str(jj)+"--")

    print(jj, np.round(xfit.mean(), 2), np.round(0.5 * (xfit.max()-xfit.min()), 2))
    print(jj, np.round(yfit.mean(), 2), np.round(0.5 * (yfit.max()-yfit.min()), 2))
    print(fit_params)

# +
import numpy as np
import matplotlib.pyplot as plt

# Example fake log-likelihood grid (Gaussian for demo)
theta1_vals = np.linspace(-3, 3, 200)
theta2_vals = np.linspace(-2, 2, 200)
T1, T2 = np.meshgrid(theta1_vals, theta2_vals, indexing="ij")

# 2D Gaussian likelihood
logL = -0.5 * (T1**2 / 1.0**2 + T2**2 / 0.5**2)

# Δχ² relative to maximum
logL_max = np.max(logL)
dchi2 = -2 * (logL - logL_max)

# --- 2D 68% region (Δχ²=2.30) ---
contour_level = 2.30
mask_2d_68 = dchi2 <= contour_level

# --- Profile likelihoods (min over the other axis) ---
profile1 = np.min(dchi2, axis=1)   # profile over θ2
profile2 = np.min(dchi2, axis=0)   # profile over θ1

# 1σ intervals (Δχ² <= 1)
theta1_interval = (theta1_vals[profile1 <= 1].min(),
                   theta1_vals[profile1 <= 1].max())
theta2_interval = (theta2_vals[profile2 <= 1].min(),
                   theta2_vals[profile2 <= 1].max())

print("68% CI for θ1:", theta1_interval)
print("68% CI for θ2:", theta2_interval)

# --- Plot ---
fig, ax = plt.subplots()

# Filled 2D 68% region
ax.contourf(T1, T2, dchi2, levels=[0, contour_level], colors=["#1f77b4"], alpha=0.3)

# Contour line
ax.contour(T1, T2, dchi2, levels=[contour_level], colors="k")

# Mark maximum likelihood
imax = np.unravel_index(np.argmin(dchi2), dchi2.shape)
ax.plot(theta1_vals[imax[0]], theta2_vals[imax[1]], "rx", label="Best fit")

# Overlay 1D intervals (as lines)
ax.axvline(theta1_interval[0], color="r", ls="--")
ax.axvline(theta1_interval[1], color="r", ls="--")
ax.axhline(theta2_interval[0], color="r", ls="--")
ax.axhline(theta2_interval[1], color="r", ls="--")

ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.legend()
# -


