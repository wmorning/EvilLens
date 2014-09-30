# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 22:53:20 2014

@author: warrenmorningstar
"""


from astropy import units, constants
from math import pi
from astropy.cosmology import angular_diameter_distance



class GravitationalLens:
    
    def __init__(self, Zd, Zs):
        
        self.Zd = Zd
        self.Zs = Zs
        self.compute_distances()
        
    def compute_distances(self):
        
        Dd = angular_diameter_distance(self.Zd)
        Ds = angular_diameter_distance(self.Zs)
        Dds = Ds - Dd
        SigmaCrit = constants.c**2 /(4*pi*constants.G) * Ds/(Dd*Dds)
        
        self.Dd = Dd
        self.Ds = Ds
        self.Dds = Dds
        self.SigmaCrit = SigmaCrit
