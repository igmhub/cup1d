# cup1d

### Cosmology using P1D - small scale clustering of the Lyman alpha forest

This repository contains some tools to perform the last steps of a cosmological analysis of the 1D power spectrum (P1D) of the Lyman alpha forest. 

p1d_archive and derived classes are containers for measured P1D from a given suite of simulations, and can be used as a starting point  to setup different emulators / interpolators. 

Test simulations can be found under sim_suites/. 

Examples for how to make several plots can be found in notebooks/.

Installing is now straightforward, and we would like to keep it this way!

Things to add in the (near) future:
 - interpolators (linear, glass...)
 - Gaussian process emulators
 - repository of P1D measurements in real data (BOSS/eBOSS/X-Shooter/UVES/HIRES) or mocks (DESI Year 1)
 - tools to generate a fake measurement given a model and a covariance matrix (MC simulation)
 
 If you would like to collaborate, please email Chris Pedersen (christian.pedersen.17@ucl.ac.uk) or Andreu Font-Ribera (a.font@ucl.ac.uk)
 
