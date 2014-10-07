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

import evillens as evil

# ======================================================================

class AnalyticSIELens(evil.GravitationalLens):
    '''
    An SIE gravitational lens object with analytic deflection angles etc.
    '''
    def __init__(self, *args, **kwargs):
        super(AnalyticSIELens, self).__init__(*args, **kwargs)
        self.sigma = 250.0
        self.q = 0.75
        return    
        
# ----------------------------------------------------------------------
 
    def build_kappa_map(self):
         
         x = sin(linspace(-self.xLength/2,self.xLength/2,self.Npixels)) * self.Dd
         y = sin(linspace(-self.yLength/2,self.yLength/2,self.Npixels)) *self.Dd
         self.image_x, self.image_y = meshgrid(x,y)
         
         Sigma_ellipsoid = self.sigma**2/(2*constants.G *sqrt(self.q*image_x**2+image_y**2))
         self.kappa = units.Quantity.to(Sigma_ellipsoid,units.solMass/units.Mpc**2) / self.SigmaCrit
         
         return
         
# ----------------------------------------------------------------------
        
    def deflect(self):
        
        x = sin(linspace(-self.xLength/2,self.xLength/2,self.Npixels)) * self.Dd /(1*units.Mpc)   
        y = sin(linspace(-self.yLength/2,self.yLength/2,self.Npixels)) * self.Dd /(1*units.Mpc) 
        image_x, image_y = meshgrid(x,y)              
        image_theta = arctan2(image_y,image_x)      

        self.alpha_x = (self.q/sqrt(1-self.q**2))*arctan(sqrt((1-self.q**2)/(self.q**2*cos(image_theta)**2+sin(image_theta)**2))*cos(image_theta))
        self.alpha_y = (self.q/sqrt(1-self.q**2))*arctanh(sqrt((1-self.q**2)/(self.q**2*cos(image_theta)**2+sin(image_theta)**2))*sin(image_theta))

        return

# ----------------------------------------------------------------------

    #def raytrace(source_image):

# ----------------------------------------------------------------------

if __name__ == '__main__':

    SIElens = evil.AnalyticSIELens(0.4,1.5)
    
    SIElens.deflect()
    print SIElens.alpha_x
        
# ======================================================================
