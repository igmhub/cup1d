import numpy as np
import json


class ArchiveP1D(object):
    """Book-keeping of flux P1D measured in a suite of simulations."""

    def __init__(self,basedir=None):
        """Constructor base class to archive P1D from simulations"""

        self.basedir=basedir
        print('Inside Base ArchiveP1D constructor',basedir)



