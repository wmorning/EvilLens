# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 22:53:20 2014

@author: warrenmorningstar
"""
# ======================================================================

from astropy import units, constants
from math import pi
from astropy.cosmology import FlatLambdaCDM
from numpy import linspace, meshgrid, arctan2, sin, cos, sqrt, arctanh, arctan
# from astropy.modeling import models

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

        # Define cosmology and compute distances, critical density.
        self.cosmological = FlatLambdaCDM(H0=70, Om0=0.3)
        self.compute_distances()
        
        return

# ----------------------------------------------------------------------
        
    def compute_distances(self):

        Dd = self.cosmological.angular_diameter_distance(self.Zd)
        Ds = self.cosmological.angular_diameter_distance(self.Zs)
        Dds = self.cosmological.angular_diameter_distance_z1z2(self.Zd,self.Zs)
        # PJM: See docs at http://docs.astropy.org/en/latest/api/astropy.cosmology.FLRW.html#astropy.cosmology.FLRW.angular_diameter_distance_z1z2
        SigmaCrit = constants.c**2 /(4*pi*constants.G) * Ds/(Dd*Dds)
        
        self.Dd = Dd
        self.Ds = Ds 
        self.Dds = Dds 
        self.SigmaCrit = units.Quantity.to(SigmaCrit,units.solMass/units.Mpc**2)
        
        return
        
 
# ----------------------------------------------------------------------
 
    def build_kappa_map(self, q = 0.5, sigmaSIE = 1.0 *units.km/units.s):
         
         x = sin(linspace(-self.xLength/2,self.xLength/2,self.Npixels)) * self.Dd
         y = sin(linspace(-self.yLength/2,self.yLength/2,self.Npixels)) *self.Dd
         image_x, image_y = meshgrid(x,y)
         
         Sigma_ellipsoid = sigmaSIE**2/(2*constants.G *sqrt(q*image_x**2+image_y**2))
         kappa = units.Quantity.to(Sigma_ellipsoid,units.solMass/units.Mpc**2) / self.SigmaCrit
         
         self.image_x = image_x
         self.image_y = image_y
         self.kappa = kappa     
         
         
         return
# ----------------------------------------------------------------------
       
     #def read_kappa_from(fitsfile):
       
# ----------------------------------------------------------------------
    
#    def deflect(self):
        

#        return
    
# ----------------------------------------------------------------------    
    
    #def realize(self, Nx, Ny, pixelscale):    
        
# ----------------------------------------------------------------------
        
    def deflect_analytic_SIE(self, Length_x=0.01, Length_y=0.01 , q=0.5):  #q is axis ratio
        
#length_x and Length_y are angular size of the lens in the image plane (in radians)
# q is axis ratio ---> q = 1-\epsilon <1
# s is core radius in Mpc. -----> add this parameter and use eq B.38 for deflection
      
        x = sin(linspace(-self.xLength/2,self.xLength/2,self.Npixels)) * self.Dd /(1*units.Mpc)   
        y = sin(linspace(-self.yLength/2,self.yLength/2,self.Npixels)) * self.Dd /(1*units.Mpc) 
        image_x, image_y = meshgrid(x,y)              
        image_theta = arctan2(image_y,image_x)      

        alpha_x = (q/sqrt(1-q**(2)))*arctan(sqrt((1-q**2)/(q**2*cos(image_theta)**2+sin(image_theta)**2))*cos(image_theta))
        alpha_y = (q/sqrt(1-q**2))*arctanh(sqrt((1-q**2)/(q**2*cos(image_theta)**2+sin(image_theta)**2))*sin(image_theta))

        self.alpha_x = alpha_x
        self.alpha_y = alpha_y


        
        return

# ----------------------------------------------------------------------

    #def raytrace(source_image):

# ----------------------------------------------------------------------

if __name__ == '__main__':

    lens = GravitationalLens(0.4,1.5)
    
    print "Difference in angular diameter distances: ",lens.Ds - lens.Dd
    print "  cf. Dds = ", lens.Dds
    print "Critical density = ",lens.SigmaCrit

    # PJM: Questions/suggestions: 
    # 1) Is SigmaCrit correct? I see units of kg/m, not kg/m^2!
    # 2) Define a units system in terms of Msun and Mpc and use that
        
# ======================================================================
