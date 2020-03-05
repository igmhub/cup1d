import numpy as np
import json


class ArchiveP1D(object):
    """Book-keeping of flux P1D measured in a suite of simulations."""

    def __init__(self,basedir=None):
        """Load arxiv from base sim directory"""

        self.basedir=basedir

