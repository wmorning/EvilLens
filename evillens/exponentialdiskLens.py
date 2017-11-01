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
from scipy.interpolate import interp1d

# ======================================================================

class ExponentialDiskLens(evil.GravitationalLens):
    '''
    An Exponential disk lens (or chameleon imitating an exponential
    disk lens), used for lower mass edge-on spiral galaxy lenses.
    
    Lens model is specified by the following parameters
    
    - b:         Deflection scale (amplitude)
    
    - q:         axis ratio
    
    - R_s:       Scale Radius
    
    - centroid:  coordinates of the lens center
    
    - angle:     angle of the major axis of the lens
    
    '''
    def __init__(self, *args, **kwargs):
        super(ExponentialDiskLens, self).__init__(*args, **kwargs)
        self.b = 1.
        self.q = 1.
        self.R_s = 0.1
        self.centroid = [0.01,0.01]              
        self.angle = 0.0
        self.n = 1
        return    
        
    def Build_kappa_map(self,b,q,R_s,centroid,angle):
        '''
        Set the parameters of the lens model, and build the convergence
        map.
        '''
        self.b = b
        self.q = q
        self.R_s = R_s
        self.centroid=centroid
        self.angle=angle + np.pi/2.  # set angle relative to y axis
        
        # set to chameleon parameters
        self.alpha_chm = pow(10,-0.739-0.527*(self.n-2.03)-0.012*pow(self.n-2.03,2)-0.008*pow(self.n-2.03,3))
        self.R0_chm = pow(10,0.078-0.184*(self.n-1.15)+0.473*pow(self.n-1.15,2)-0.079*pow(self.n-1.5,3))*self.R_s
        
        
        x1p = self.image_x - self.centroid[0]
        x2p = self.image_y - self.centroid[1]
        
        x1 = np.cos(-self.angle)*x1p -np.sin(-self.angle)*x2p
        x2 = np.sin(-self.angle)*x1p +np.cos(-self.angle)*x2p
        self.kappa = b / 2. /self.q / np.sqrt(self.R0_chm**2+x1**2+(x2/self.q)**2)
        self.kappa -=  b / 2. /self.q / np.sqrt((self.R0_chm/self.alpha_chm)**2+x1**2+(x2/self.q)**2)
        
        
        
    def deflect(self):
        '''
        '''
        
        x1p = self.image_x - self.centroid[0]
        x2p = self.image_y - self.centroid[1]
        
        x1 = np.cos(-self.angle)*x1p -np.sin(-self.angle)*x2p
        x2 = np.sin(-self.angle)*x1p +np.cos(-self.angle)*x2p
        
        alphaXp1 = self.b/pow(1-pow(self.q,2),0.5)* np.arctan(pow(1-pow(self.q,2),0.5)*x1/(np.sqrt(pow(self.q,2)*(pow(self.R0_chm,2)+pow(x1,2))+pow(x2,2))+pow(self.q,2)*self.R0_chm))
        alphaYp1 = self.b/pow(1-pow(self.q,2),0.5)*np.arctanh(pow(1-pow(self.q,2),0.5)*x2/(np.sqrt(pow(self.q,2)*(pow(self.R0_chm,2)+pow(x1,2))+pow(x2,2))+pow(self.q,2)+self.R0_chm))
        
        alphaXp2 = self.b/pow(1-pow(self.q,2),0.5)* np.arctan(pow(1-pow(self.q,2),0.5)*x1/(np.sqrt(pow(self.q,2)*(pow(self.R0_chm/self.alpha_chm,2)+pow(x1,2))+pow(x2,2))+self.R0_chm/self.alpha_chm))
        alphaYp2 = self.b/pow(1-pow(self.q,2),0.5)*np.arctanh(pow(1-pow(self.q,2),0.5)*x2/(np.sqrt(pow(self.q,2)*(pow(self.R0_chm/self.alpha_chm,2)+pow(x1,2))+pow(x2,2))+pow(self.q,2)*self.R0_chm/self.alpha_chm))
        
        alphaXp = alphaXp1 - alphaXp2
        alphaYp = alphaYp1 - alphaYp2
        
        self.alpha_x = np.cos(self.angle)*alphaXp -np.sin(self.angle)*alphaYp
        self.alpha_y = np.sin(self.angle)*alphaXp + np.cos(self.angle)*alphaYp