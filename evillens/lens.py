# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 22:53:20 2014

@author: warrenmorningstar
"""
# ======================================================================

from astropy import units, constants
from math import pi
from astropy.cosmology import FlatLambdaCDM
# from astropy.modeling import models

# ======================================================================

class GravitationalLens(object):
    '''
    An object class describing a gravitational lens system.
    '''
    def __init__(self, Zd, Zs):
        
        self.Zd = Zd
        self.Zs = Zs
        self.Dd = None
        self.Ds = None
        self.Dds = None
        self.SigmaCrit = None

        # Define cosmology and compute distances, critical density.
        self.cosmological = FlatLambdaCDM(H0=70, Om0=0.3)
        self.compute_distances()
        
        return

# ----------------------------------------------------------------------
        
    def compute_distances(self):

        Dd = self.cosmological.angular_diameter_distance(self.Zd)
        Ds = self.cosmological.angular_diameter_distance(self.Zs)
        # Dds = Ds - Dd # PJM: this is not correct
        Dds = self.cosmological.angular_diameter_distance_z1z2(self.Zd,self.Zs)
        # PJM: See docs at http://docs.astropy.org/en/latest/api/astropy.cosmology.FLRW.html#astropy.cosmology.FLRW.angular_diameter_distance_z1z2
        SigmaCrit = constants.c**2 /(4*pi*constants.G) * Ds/(Dd*Dds)
<<<<<<< HEAD
        
        self.Dd = Dd
        self.Ds = Ds
        self.Dds = Dds
        self.SigmaCrit = units.Quantity.to(SigmaCrit, units.solMass/units.Mpc**2)
        

=======

        self.Dd = Dd * units.Mpc
        self.Ds = Ds * units.Mpc
        self.Dds = Dds * units.Mpc
        self.SigmaCrit = units.Quantity.decompose(SigmaCrit)
        
        return
        
# ----------------------------------------------------------------------

#    def build_from( x_length , y_length , N_Sidepoints , function):
#        return
>>>>>>> 3824161062d8c9a5f7b353b31f7f0a9038b10c07
        
# ======================================================================

if __name__ == '__main__':

    lens = GravitationalLens(0.4,1.5)
    
    print "Difference in angular diameter distances: ",lens.Ds - lens.Dd
    print "  cf. Dds = ", lens.Dds
    print "Critical density = ",lens.SigmaCrit

    # PJM: Questions/suggestions: 
    # 1) Is SigmaCrit correct? I see units of kg/m, not kg/m^2!
    # 2) Define a units system in terms of Msun and Mpc and use that
        
# ======================================================================
