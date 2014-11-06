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
        self.centroid = [0.01,0.01]  #WRM: centroid offsets center of map from
                                    #     origin to avoid divergence at a pixel
        
        return    
        
# ----------------------------------------------------------------------
 
    def build_kappa_map(self):
         
         # First compute Einstein radius, in arcsec:
         b = 4*np.pi *(self.sigma/constants.c)**2*(np.sqrt(1-self.q**2)/np.arcsin(np.sqrt(1-self.q**2)))*self.Dds/self.Ds
         self.b = b.decompose()*(3600.0*180.0/np.pi)
         
         # Now compute kappa using lens equation from Saas Fe.
         self.kappa = self.b.value /(2.0*np.sqrt(self.q*(self.x-self.centroid[0])**2+(self.y-self.centroid[1])**2))

         return
         
# ----------------------------------------------------------------------
        
    def deflect(self):
        
        # First convert image coordinates to angle from semimajor axis.
        #image_theta = np.arctan2(self.image_y-self.centroid[1],self.image_x-self.centroid[0]) # check...

        # Deflect analytically.  This is old way, but it should be the same.
        #self.alpha_x = (self.b.value/np.sqrt(1-self.q**2))*np.arctan(np.sqrt((1-self.q**2)/(self.q**2*np.cos(image_theta)**2+np.sin(image_theta)**2))*np.cos(image_theta))
        #self.alpha_y = (self.b.value/np.sqrt(1-self.q**2))*np.arctanh(np.sqrt((1-self.q**2)/(self.q**2*np.cos(image_theta)**2+np.sin(image_theta)**2))*np.sin(image_theta))
        
        # Deflect analytically using equation from Saas Fe
        self.alpha_x = (self.b.value/np.sqrt(1-self.q**2)) * np.arctan((self.image_x-self.centroid[0])*np.sqrt(1-self.q**2)/np.sqrt(self.q**2*(self.image_x-self.centroid[0])**2+(self.image_y-self.centroid[1])**2))        
        self.alpha_y = (self.b.value/np.sqrt(1-self.q**2)) * np.arctanh((self.image_y-self.centroid[1])*np.sqrt(1-self.q**2)/np.sqrt(self.q**2*(self.image_x-self.centroid[0])**2+(self.image_y-self.centroid[1])**2))
        return

# ======================================================================

if __name__ == '__main__':

    SIElens = evil.AnalyticSIELens(0.4,1.5, q=0.75, centroid=[.01,.01])
    
    SIElens.deflect()
    print SIElens.alpha_x
    
    SIElens.build_kappa_map()
    
        
# ======================================================================
