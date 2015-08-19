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

    def build_kappa_map(self, M, c, q=1, centroid = [0,0]):
        if q!=1:
            raise Exception('Cant do elliptical NWF lenses yet \n')
            self.e = 1 - q   #conventional ellpiticity = 1 - b / a
            self.epsilon = (2*(1-self.e)-(1-self.e)**2)/(2-2*(1-self.e)+(1-self.e)**2)
            self.a1 = 1-self.epsilon  #re-define coordinate system
            self.a2 = 1+self.epsilon  #using new elliptical coordinates
            self.x1 = np.sqrt(self.a1)*self.x 
            self.x2 = np.sqrt(self.a2)*self.y
            self.xe = np.sqrt(self.x1**2+self.x2**2)
            self.phi_e = np.arctan(self.x2/self.x1)
        self.M = M *units.solMass
        self.c = c
        self.q = q 
        self.centroid = centroid
        self.delta_c = 200/3 * c**3/(np.log(1+c)-c/(1+c))
        self.rho_c = (3.0/8.0 * self.cosmological.H(self.Zd)**2/(8.0*np.pi*constants.G)).to(units.solMass/units.Mpc**3)
        self.r200 = ((self.M/(800*np.pi/3*self.rho_c))**(1.0/3.0)).to(units.Mpc)
        self.rs = np.arctan(self.r200 / self.c / self.Dd).to(units.arcsec).value
        self.kappa_c = 2*(self.rho_c*self.delta_c*self.r200/self.c/self.SigmaCrit).decompose()
        
        r_norm = np.sqrt((self.x-self.centroid[0])**2+(self.y-self.centroid[1])**2)/self.rs
        
        Fx = np.zeros(self.x.shape,float)
        Fx[np.where(r_norm < 1 )] = 1.0/(r_norm[np.where(r_norm<1)]**2-1.0)*(1.0-(np.arccosh(1.0/r_norm[np.where(r_norm<1)])/np.sqrt(1.0-r_norm[np.where(r_norm<1)]**2)))
        Fx[np.where(r_norm == 1)] = 1.0/3.0
        Fx[np.where(r_norm > 1 )] = 1.0/(r_norm[np.where(r_norm>1)]**2-1.0)*(1.0-(np.arccos( 1.0/r_norm[np.where(r_norm>1)])/np.sqrt(r_norm[np.where(r_norm>1)]**2-1.0)))
#        for i in range(len(self.x[:,0])):
#            for j in range(len(self.x[0,:])):
#                if r_norm[i,j]<1:
#                    Fx[i,j] = 1.0/(r_norm[i,j]**2-1)*(1-np.arccosh(1.0/r_norm[i,j])/np.sqrt(1-r_norm[i,j]**2))
#                elif r_norm[i,j]==1:
#                    Fx[i,j] = 1.0/3.0
#                else:
#                    Fx[i,j] = 1.0/(r_norm[i,j]**2-1)*(1-np.arccos(1/r_norm[i,j])/np.sqrt(r_norm[i,j]**2-1))
        
        self.kappa = 2.0*self.kappa_c*Fx
        
        return
        
# ---------------------------------------------------------------------------        
    def deflect(self):
        r_norm = np.sqrt((self.image_x-self.centroid[0])**2+(self.image_y-self.centroid[1])**2)/self.rs
        self.r_norm=r_norm
        gx = np.zeros(self.image_x.shape)
        gx[np.where(r_norm < 1) ] = np.log(r_norm[np.where(r_norm<1)]/2.0)+np.arccosh(1.0/r_norm[np.where(r_norm<1)])/np.sqrt(1-r_norm[np.where(r_norm<1)]**2)
        gx[np.where(r_norm == 1)] = 1.0+np.log(0.5)
        gx[np.where(r_norm > 1) ] = np.log(r_norm[np.where(r_norm>1)]/2.0)+np.arccos(1.0/r_norm[np.where(r_norm>1)])/np.sqrt(r_norm[np.where(r_norm>1)]**2-1)      
#        for i in range(len(self.image_x[:,0])):
#            for j in range(len(self.image_x[0,:])):
#                if r_norm[i,j]<1:
#                    gx[i,j] = np.log(r_norm[i,j]/2.0)+np.arccosh(1.0/r_norm[i,j])/np.sqrt(1-r_norm[i,j]**2)
#                elif r_norm[i,j] ==1:
#                    gx[i,j]=1.0+np.log(0.5)
#                else:
#                    gx[i,j] = np.log(r_norm[i,j]/2.0)+np.arccos(1.0/r_norm[i,j])/np.sqrt(r_norm[i,j]**2-1)
        self.alpha_x = (self.image_x-self.centroid[0])*(4.0*self.kappa_c*gx)/r_norm**2
        self.alpha_y = (self.image_y-self.centroid[1])*(4.0*self.kappa_c*gx)/r_norm**2
        