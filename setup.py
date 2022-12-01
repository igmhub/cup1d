#!/usr/bin/env python

from setuptools import setup, find_packages

description = "Cosmology using P1D - small scale clustering of the Lyman alpha forest"
version="0.1.0"

setup(name="cup1d",
    version=version,
    description=description,
    url="https://github.com/igmhub/cup1d",
    author="Andreu Font-Ribera et al.",
    author_email="afont@ifae.es",
    packages=find_packages(),
    )
