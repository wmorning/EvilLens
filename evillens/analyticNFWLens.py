# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:10:24 2015

@author: wmorning
"""

# ===========================================================================

from astropy import units, constants
import numpy as np
import evillens as evil
import scipy.special as sp
import matplotlib.pyplot as plt

# ===========================================================================

class AnalyticNFWLens(evil.GravitationalLens):
    '''
    An analytic elliptical NFW lens profile with analytic deflection angles etc.
    for use in simulating DM substructures or larger structures.
    '''
    def __init__(self,*args,**kwargs):
        super(AnalyticNFWLens, self ).__init__(*args, **kwargs)
        self.M = 0
        self.a = 0
        self.centroid = [0,0]
        return
# ---------------------------------------------------------------------------

    def build_kappa_map(self, r_c, v_c, q=1, centroid = [0,0]):
        if q!=1:
            raise Exception('Cant do elliptical NWF lenses yet \n')
        self.r_c = r_c *units.kpc
        self.v_c = v_c*units.km/units.s
        self.q = q 
        self.centroid = centroid
        self.theta_c = np.arctan(self.r_c/self.Dd).to(units.arcsec).value
        self.rho_c = (3.0/8.0*self.v_c**2/self.r_c**2/constants.G).decompose()
        
        self.kappa_c = (self.rho_c*self.r_c/self.SigmaCrit).decompose()
        
        r_norm = np.sqrt((self.x-self.centroid[0])**2+(self.y-self.centroid[1])**2)/self.theta_c
        
        Fx = np.zeros(self.x.shape,float)
        for i in range(len(self.x[:,0])):
            for j in range(len(self.x[0,:])):
                if r_norm[i,j]<1:
                    Fx[i,j] = 1.0/(r_norm[i,j]**2-1)*(1-np.arccosh(1.0/r_norm[i,j])/np.sqrt(1-r_norm[i,j]**2))
                elif r_norm[i,j]==1:
                    Fx[i,j] = 1.0/3.0
                else:
                    Fx[i,j] = 1.0/(r_norm[i,j]**2-1)*(1-np.arccos(1/r_norm[i,j])/np.sqrt(r_norm[i,j]**2-1))
        
        self.kappa = 2.0*self.kappa_c*Fx
        
        return
        
# ---------------------------------------------------------------------------        
    def deflect(self):
        r_norm = np.sqrt((self.image_x-self.centroid[0])**2+(self.image_y-self.centroid[1])**2)/self.theta_c
        self.r_norm=r_norm
        gx = np.zeros(self.image_x.shape)
        for i in range(len(self.image_x[:,0])):
            for j in range(len(self.image_x[0,:])):
                if r_norm[i,j]<1:
                    gx[i,j] = np.log(r_norm[i,j]/2.0)+np.arccosh(1.0/r_norm[i,j])/np.sqrt(1-r_norm[i,j]**2)
                elif r_norm[i,j] ==1:
                    gx[i,j]=1.0+np.log(0.5)
                else:
                    gx[i,j] = np.log(r_norm[i,j]/2.0)+np.arccos(1.0/r_norm[i,j])/np.sqrt(r_norm[i,j]**2-1)
        self.alpha_x = (self.image_x-self.centroid[0])*(4.0*self.kappa_c*gx)/r_norm**2
        self.alpha_y = (self.image_y-self.centroid[1])*(4.0*self.kappa_c*gx)/r_norm**2
        