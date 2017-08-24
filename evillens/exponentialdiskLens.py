# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 22:53:20 2014

@author: warrenmorningstar
"""
# ======================================================================

from astropy import units, constants
from astropy.cosmology import FlatLambdaCDM

import numpy as np
import evillens as evil
from scipy.interpolate import interp1d

# ======================================================================

class ExponentialDiskLens(evil.GravitationalLens):
    '''
    An Exponential disk lens (or chameleon imitating an exponential
    disk lens), used for lower mass edge-on spiral galaxy lenses.
    
    Lens model is specified by the following parameters
    
    - b:         Deflection scale (amplitude)
    
    - q:         axis ratio
    
    - R_s:       Scale Radius
    
    - centroid:  coordinates of the lens center
    
    - angle:     angle of the major axis of the lens
    
    '''
    def __init__(self, *args, **kwargs):
        super(ExponentialDiskLens, self).__init__(*args, **kwargs)
        self.b = 1.
        self.q = 1.
        self.R_s = 0.1
        self.centroid = [0.01,0.01]              
        self.angle = 0.0
        return    
        
        