#!/usr/bin/env python

from setuptools import setup, find_packages

description = "Cosmology using P1D - small scale clustering of the Lyman alpha forest"
version="0.1"

setup(name="cup1d",
    version=version,
    description=description,
    url="https://github.com/igmhub/cup1d",
    author="Chris Pedersen, Andreu Font-Ribera et al.",
    author_email="christian.pedersen.17@ucl.ac.uk",
    packages=find_packages(),
    )
