# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:49:16 2015

@author: wmorning
"""

# ===========================================================================

from astropy import units
import numpy as np
import evillens as evil
import scipy.special as sp
import matplotlib.pyplot as plt

# ===========================================================================

class AnalyticPseudoJaffeLens(evil.GravitationalLens):
    '''
    An analytic Pseudo-Jaffe lens profile with analytic deflection angles etc.
    for use in simulating DM substructures.
    '''
    def __init__(self,*args,**kwargs):
        super(AnalyticPseudoJaffeLens, self ).__init__(*args, **kwargs)
        self.M = 0
        self.a = 0
        self.centroid = [0,0]
        return
# ---------------------------------------------------------------------------
    def build_kappa_map(self,M=0,a=0.1,centroid=[0,0],n=4,GAMMA=2):
        '''
        Create kappa map from Pseudo-Jaffe profile
        -M is total mass of the subhalo
        -n is outer exponent (defaults to 4, should be greater than 3)
        -gamma is cusp exponent        
        -a is break radius in units of arcsec
        -centroid is coordinates of lens center
        '''
        self.M = float(M)*units.solMass
        self.a = a
        self.centroid = centroid
        self.n_outer = n
        self.gamma = GAMMA
        if self.n_outer ==3:
            raise Exception("this profile doesn't work for n=3 \n")
        #first compute convergence scale kappa_0.        
        self.kappa_0 = (self.M/(2.0*np.pi*(self.a/(3600.0*180.0/np.pi)*self.Dd)**2*sp.beta((self.n_outer-3)/2.0,(3-self.gamma)/2.0))/self.SigmaCrit).value
        xi = np.sqrt((self.x-self.centroid[0])**2+(self.y-self.centroid[1])**2)/self.a
        self.kappa = self.kappa_0*sp.beta((self.n_outer-1)/2.0,1.0/2.0)*(1+xi**2)**((1-self.n_outer)/2.0)*sp.hyp2f1((self.n_outer-1)/2.0,self.gamma/2.0,self.n_outer/2.0,1/(1+xi**2))
        
        return
# ---------------------------------------------------------------------------
    def deflect(self):
        xi =np.sqrt((self.image_x-self.centroid[0])**2+(self.image_y-self.centroid[1])**2)/self.a 
        alpha = 2*self.kappa_0*self.a/xi * (sp.beta((self.n_outer-3)/2.0,(3-self.gamma)/2.0) \
                - sp.beta((self.n_outer-3)/2.0, 3.0/2.0) * (1+xi**2)**((3-self.n_outer)/2.0)\
                * sp.hyp2f1((self.n_outer-3)/2.0,self.gamma/2.0,self.n_outer/2.0,1/(1+xi**2)))
                
        self.alpha_x = alpha*(self.image_x-self.centroid[0])/np.sqrt((self.image_x-self.centroid[0])**2+(self.image_y-self.centroid[1])**2)
        self.alpha_y = alpha*(self.image_y-self.centroid[1])/np.sqrt((self.image_x-self.centroid[0])**2+(self.image_y-self.centroid[1])**2)
        
# ======================================================================

if __name__ == '__main__':

    PSJlens = evil.AnalyticPseudoJaffeLens(0.5,3.0)
    PSJlens.setup_grid(NX=100,NY=100,pixscale=0.01)
    PSJlens.build_kappa_map(1.0*10**8,0.1,[0.0005,0.0005])
    
    plt.imshow(np.log10(PSJlens.kappa), cmap='cubehelix')
    
        
# ======================================================================
