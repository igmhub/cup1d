import numpy as np
import json
from . import p1d_archive


class ExampleArchiveP1D(p1d_archive.ArchiveP1D):
    """Book-keeping of flux P1D measured in a suite of Example simulations."""

    def __init__(self,basedir=None):
        """Load arxiv from directory containing Example simulations."""

        print('ExampleArchiveP1D will call constructor of ArchiveP1D')
        ArchiveP1D.__init__(self,basedir)


