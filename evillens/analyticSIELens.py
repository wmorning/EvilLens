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

# ======================================================================

class AnalyticSIELens(evil.GravitationalLens):
    '''
    An SIE gravitational lens object with analytic deflection angles etc.
    '''
    def __init__(self, *args, **kwargs):
        super(AnalyticSIELens, self).__init__(*args, **kwargs)
        self.sigma = 200.0 * units.km/units.s
        self.q = 0.75
        return    
        
# ----------------------------------------------------------------------
 
    def build_kappa_map(self):
         
         x = np.sin(np.linspace(-self.xLength/2,self.xLength/2,self.Npixels)) * self.Dd
         y = np.sin(np.linspace(-self.yLength/2,self.yLength/2,self.Npixels)) *self.Dd
         self.image_x, self.image_y = np.meshgrid(x,y)
         
         Sigma_ellipsoid = self.sigma**2/(2*constants.G *np.sqrt(self.q*self.image_x**2+self.image_y**2))
         self.kappa = units.Quantity.to(Sigma_ellipsoid,units.solMass/units.Mpc**2) / self.SigmaCrit
         
         return
         
# ----------------------------------------------------------------------
        
    def deflect(self):
        
        x = np.sin(np.linspace(-self.xLength/2,self.xLength/2,self.Npixels)) * self.Dd /(1*units.Mpc)   
        y = np.sin(np.linspace(-self.yLength/2,self.yLength/2,self.Npixels)) * self.Dd /(1*units.Mpc) 
        image_x, image_y = np.meshgrid(x,y)              
        image_theta = np.arctan2(image_y,image_x)      

        self.alpha_x = (self.q/np.sqrt(1-self.q**2))*np.arctan(np.sqrt((1-self.q**2)/(self.q**2*np.cos(image_theta)**2+np.sin(image_theta)**2))*np.cos(image_theta))
        self.alpha_y = (self.q/np.sqrt(1-self.q**2))*np.arctanh(np.sqrt((1-self.q**2)/(self.q**2*np.cos(image_theta)**2+np.sin(image_theta)**2))*np.sin(image_theta))

        return

# ----------------------------------------------------------------------



# ----------------------------------------------------------------------

    #def raytrace(source_image):

# ======================================================================

if __name__ == '__main__':

    SIElens = evil.AnalyticSIELens(0.4,1.5)
    
    SIElens.deflect()
    print SIElens.alpha_x
    
    SIElens.build_kappa_map()
        
# ======================================================================
