# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 22:53:20 2014

@author: warrenmorningstar
"""
# ======================================================================

from astropy import units, constants
from math import pi
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import numpy as np

# ======================================================================

class GravitationalLens(object):
    '''
    An object class describing a gravitational lens system.
    '''
    def __init__(self, Zd, Zs, Npixels = 5, xLength = .01, yLength = .01):
        
        self.Zd = Zd
        self.Zs = Zs
        self.Npixels = Npixels
        self.xLength = xLength
        self.yLength = yLength
        self.Dd = None
        self.Ds = None
        self.Dds = None
        self.SigmaCrit = None

        self.cosmological = FlatLambdaCDM(H0=70, Om0=0.3)
        self.compute_distances()
        
        return

# ----------------------------------------------------------------------
        
    def compute_distances(self):

        Dd = self.cosmological.angular_diameter_distance(self.Zd)
        Ds = self.cosmological.angular_diameter_distance(self.Zs)
        Dds = self.cosmological.angular_diameter_distance_z1z2(self.Zd,self.Zs)
        SigmaCrit = constants.c**2 /(4*pi*constants.G) * Ds/(Dd*Dds)
        
        self.Dd = Dd
        self.Ds = Ds 
        self.Dds = Dds 
        self.SigmaCrit = units.Quantity.to(SigmaCrit,units.solMass/units.Mpc**2)
        
        return
 
# ----------------------------------------------------------------------
 
    def build_kappa_map(self):
        raise Exception("Can't build a kappa map.\n")
        return
        
# ----------------------------------------------------------------------
       
    def read_kappa_from(self,fitsfile):
        raise Exception("Can't read in kappa maps yet.\n") 
        #self.kappa = Table.read(fitsfile)
        return
        
# ----------------------------------------------------------------------
    
    def deflect(self):
        raise Exception("Can't compute deflections yet.\n")  
        return
    
# ----------------------------------------------------------------------    
    
    def realize(self, Nx, Ny, pixelscale):    
        raise Exception("Can't plot anything yet.\n")  
        return

# ----------------------------------------------------------------------

    def write_kappa_to_fits(self,fitsfile):
        raise Exception("Can't write kappa map to fits yet.\n")
        #self.kappa.write( fitsfile )         
        return
        
# ----------------------------------------------------------------------

    def raytrace(source_image):
        raise Exception("Can't do raytracing yet.\n")  
        return

# ======================================================================

if __name__ == '__main__':

    lens = GravitationalLens(0.4,1.5)
    
    print "Difference in angular diameter distances: ",lens.Ds - lens.Dd
    print "  cf. Dds = ", lens.Dds
    print "Critical density = ",lens.SigmaCrit

        
# ======================================================================
