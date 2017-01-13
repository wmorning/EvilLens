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

class PowerKappa(evil.GravitationalLens):
    '''
    An Elliptical Power-law gravitational lens object with fast numerically calculated
    deflection angles from Barkana 1998.
    
    Lens model is specified by the following parameters
    
    - logM:      base-10 log of the mass
    
    - q:         axis ratio
    
    - Gamma:     Power-law slope (between 0 and 2)
    
    - centroid:  coordinates of the lens center
    
    - angle:     angle of the major axis of the lens
    
    - r_c:       core radius (in radians)
    '''
    def __init__(self, *args, **kwargs):
        super(PowerKappa, self).__init__(*args, **kwargs)
        self.logM = 1.15
        self.q = 0.75
        self.Gamma = 1.0
        self.centroid = [0.01,0.01] 
        self.r_c = 0.0               
        self.angle = 0.0
        return    
        
# ----------------------------------------------------------------------
    def build_kappa_map(self,logM=None,q=None,Gamma=None,centroid=None,angle=None,r_c=None):
        '''
        Build the density map, and adjust the parameters if specified
        '''
        if logM is not None:
            self.logM = logM
        
        if q is not None:
            self.q = q
            
        if Gamma is not None:
            self.Gamma = Gamma
        
        if centroid is not None:
            self.centroid = centroid
            
        if angle is not None:
            self.angle = angle
            
        if r_c is not None:
            self.r_c = r_c
            
        # define additional parameters for Barkana 1998    
        self.Mass = 10**(10*self.logM) * units.solMass
        self.sigma = np.sqrt(self.Mass / np.sqrt(self.q)*constants.G/(10.*units.kpc)/np.pi).to(units.km/units.s) 
        gam = self.Gamma / 2.
        R_mass = 10e03 / self.Dd.to(units.pc).value
        SigmaCrit = (self.SigmaCrit * self.Dd **2).decompose()
        Q = (self.Mass / (2*np.pi*SigmaCrit * (self.r_c**(2*(1-gam))-(R_mass**2+self.r_c**2)**(1-gam))/(2.0*(gam-1.0)) ) / (self.q)).decompose()
        Q = Q.value 
        self.gam = gam
        self.Q = Q
        
        # Center coordinates on lens.
        x = (self.x - self.centroid[0]) / 3600.0 / 180.0 * np.pi
        y = (self.y - self.centroid[1]) / 3600.0 / 180.0 * np.pi
        
        # rotate to lens major axis coordinate system
        angle = self.angle + np.pi / 2.
        xprime = np.cos(angle) * x + np.sin(angle) * y
        yprime = -np.sin(angle) * x + np.cos(angle)* y
        
        # build kappa map
        rhosq = x**2+y**2/self.q**2
        self.kappa = (self.Q**2/(rhosq+self.r_c**2))**self.gam
        
        return

# ----------------------------------------------------------------------

    def deflect(self):
        
        # Center coordinates on lens.
        x = (self.image_x - self.centroid[0]) / 3600.0 / 180.0 * np.pi
        y = (self.image_y - self.centroid[1]) / 3600.0 / 180.0 * np.pi
        
        # rotate to lens major axis coordinate system
        angle = self.angle + np.pi / 2.
        xprime = np.cos(angle) * x + np.sin(angle) * y
        yprime = -np.sin(angle) * x + np.cos(angle)* y
        
        # Get deflection angles
        n = len(x.ravel())
        alpha_x = np.empty(n)
        alpha_y = np.empty(n)
        evil._fastell.fastelldefl_array(xprime.ravel(),yprime.ravel(),self.Q,self.gam,self.q,self.r_c,alpha_x,alpha_y,n)
        
        # Want deflections in arcseconds
        alpha_x = alpha_x.reshape(self.image_x.shape) * 3600.0 * 180.0 / np.pi
        alpha_y = alpha_y.reshape(self.image_y.shape) * 3600.0 * 180.0 / np.pi
        
        # rotate back to original frame
        self.alpha_x = np.cos(angle) * alpha_x - np.sin(angle) * alpha_y
        self.alpha_y = np.sin(angle) * alpha_x + np.cos(angle) * alpha_y
        
        
        return
        
# ----------------------------------------------------------------------

    def add_subhalos(self, M , centroid , N ):
        '''
        Add N subhalos with masses M and positions given by centroid
        to you SIE lens.  Want to determine the tidal radius based on
        subhalo mass and main halo mass.
        
        For this, we want to minimize the impact on memory used in adding
        a subhalo.  What we should thus do is remember the subhalo parameters
        as part of the main lens, but not remember the kappa grids or 
        alpha grids (have them be local variables)
        '''
        
        
        if self.kappa is None:
            raise Exception('Need main lens kappa map \n')
        if self.alpha_x is None:
            raise Exception('Need main lens deflection angles \n')
        assert self.alpha_x.shape == self.image_x.shape
        assert self.alpha_y.shape == self.image_y.shape
        assert self.alpha_x.shape == self.alpha_y.shape
        assert self.x.shape == self.kappa.shape
        assert self.kappa.shape == self.y.shape
        
        
        self.Nsubhalos = N
        Msub = M*units.solMass
        # calculate tidal radius of subhalo using parameters of main halo
        EinRad_M = 4.0*np.pi*(self.sigma/constants.c)**2* self.Dds/self.Ds
        
        
        Sigma_sub = (np.sqrt(4.0/np.pi) * constants.G * Msub *self.sigma \
                    /(np.pi * EinRad_M * self.Dd))**(1.0/3.0)
        Rtidal = (Sigma_sub / self.sigma / np.sqrt(4.0/np.pi)) * EinRad_M 
        Rcore = Rtidal.decompose().value * 3600.0*180.0/np.pi
        #create subhalo object.  Have coordinates be those of main halo.
        
        subhalo = evil.AnalyticPseudoJaffeLens(self.Zd,self.Zs)
        subhalo.x = self.x
        subhalo.y = self.y
        subhalo.image_x = self.image_x
        subhalo.image_y = self.image_y  
        if N ==1:
            subhalo.build_kappa_map(Msub.value, a = Rcore , \
                centroid = centroid, n = 4, GAMMA = 2)
            subhalo.deflect()
            self.kappa += subhalo.kappa
            self.alpha_x += subhalo.alpha_x
            self.alpha_y += subhalo.alpha_y
        else:
            for i in range(N):
                subhalo.build_kappa_map(Msub[i].value, a = Rcore[i] , \
                    centroid = centroid[i], n = 4, GAMMA = 2)
                subhalo.deflect()
                self.kappa += subhalo.kappa
                self.alpha_x += subhalo.alpha_x
                self.alpha_y += subhalo.alpha_y
        self.subhalo_masses = Msub.value
        self.subhalo_Rcore = Rcore
        self.subhalo_positions = centroid
        
        return
        
# ----------------------------------------------------------------------

    def add_multipoles(self,M):
        '''
        Add angular multipoles to the lens model.
        
        Accepts a Nx2 vector specifying the N multipole moments to add.
        The 2nd order multipole is external shear.  Definitions of
        parameters are made to be compatible with Ripples (i.e. x,y 
        components rather than magnitude and angle)
        
        The x,y components contain amplitude and phase information of
        the shear/multipoles.  Note that the phase (arctan2(My,Mx))/m is 
        defined WRT the positive x-axis.
        '''
        
        M = np.array(M)
        if len(M.shape) == 1: # Compatible with 1d shear array
            M = M.reshape([1,2])
        Nmoments = M.shape[0]
                
        # Get image polar coordinates in radians
        imR     = np.sqrt((self.image_x-self.centroid[0])**2+(self.image_y-self.centroid[1])**2) / 3600.0 / 180.0 * np.pi
        imTheta = np.arctan2(self.image_y-self.centroid[1],self.image_x-self.centroid[0])
        Rs = np.pi / 180.0 / 3600.0
        GAMMA = self.Gamma
        
        for i in range(Nmoments):
            m = float(i+2)  # The first multipole is m=2 (shear)
            if m == 2:
                
                Am = M[i,0] 
                Bm = M[i,1]
                
                # calculate deflections
                alphaR     = Am * imR * np.cos(2*imTheta) + Bm * imR * np.sin(2*imTheta)
                alphaTheta = -Am * imR * np.sin(2*imTheta)+ Bm * imR * np.cos(2*imTheta)
                
                # convert back to cartesian coords
                alphaX = alphaR * np.cos(imTheta) + alphaTheta * (-np.sin(imTheta))
                alphaY = alphaR * np.sin(imTheta) + alphaTheta * (np.cos(imTheta))
                
                # subtract shear from current deflection angles, in arcsec
                # NOTE: May influence inferred direction of shear if done
                # wrong.
                self.alpha_x -= alphaX * 3600.0 * 180.0 / np.pi
                self.alpha_y -= alphaY * 3600.0 * 180.0 / np.pi
            else:  # higher order multipoles
                Am = M[i,0] 
                Bm = M[i,1]
                
                # calculate deflections
                alphaR     = (1/(GAMMA**2 - 4*GAMMA+4-m**2)) * (2-GAMMA)*(imR**(1-GAMMA)/(Rs)**(-GAMMA)) * (Am * np.cos(m*imTheta) + Bm * np.sin(m*imTheta))
                alphaTheta = (m/(GAMMA**2-4*GAMMA+4-m**2)) * (2-GAMMA)*(imR**(1-GAMMA)/(Rs)**(-GAMMA)) * (-Am * np.sin(m*imTheta) + Bm * np.cos(m*imTheta))
 
                # convert back to cartesian coords
                alphaX = alphaR * np.cos(imTheta) + alphaTheta * (-np.sin(imTheta))
                alphaY = alphaR * np.sin(imTheta) + alphaTheta * (np.cos(imTheta))
                
                # add shear to current deflection angles
                self.alpha_x -= alphaX * 3600.0 * 180.0 / np.pi
                self.alpha_y -= alphaY * 3600.0 * 180.0 / np.pi
                
                self.kappa += 0.5*(np.gradient(alphaX,self.pixscale)[1]+np.gradient(alphaY,self.pixscale)[0])
            
        self.Multipoles = M
        
        return
        
# ----------------------------------------------------------------------

    def remove_multipoles(self):
        '''
        Just in case we ever want to get rid of added multipoles.  Does 
        nearly the same thing as add_multipoles, but subtracts them from
        deflection instead.  Only works if multipoles are already there.
        '''
        if not hasattr(self,'Multipoles'):
            
            print "Lens doesn't have multipoles"
            return
        
        else:
            
            M = np.array(self.Multipoles)
            if len(M.shape) == 1:
                M = M.reshape([1,2])
            Nmoments = M.shape[0]
            Rs = 1 * np.pi / 180.0 / 3600.0
        
            # Get image polar coordinates in radians
            imR     = np.sqrt((self.image_x-self.centroid[0])**2+(self.image_y-self.centroid[1])**2) * np.pi / 3600.0 / 180.0
            imTheta = np.arctan2(self.image_y-self.centroid[1],self.image_x-self.centroid[0])
        
            for i in range(Nmoments):
                m = i+2  # The first multipole is m=2 (shear)
                if m == 2:
                    n = 2
                    
                    Am = M[i,0] 
                    Bm = M[i,1]
                
                    # calculate deflections
                    alphaR     = Am * imR * np.cos(2*imTheta) + Bm * imR * np.sin(2*imTheta)
                    alphaTheta = -Am * imR * np.sin(2*imTheta)+ Bm * imR * np.cos(2*imTheta)
                
                    # convert back to cartesian coords
                    alphaX = alphaR * np.cos(imTheta) + alphaTheta * (-np.sin(imTheta))
                    alphaY = alphaR * np.sin(imTheta) + alphaTheta * (np.cos(imTheta))
                
                    # add shear to current deflection angles, in arcsec
                    self.alpha_x += alphaX * 3600 * 180 / np.pi
                    self.alpha_y += alphaY * 3600 * 180 / np.pi
                else:
                    Am = M[i,0] 
                    Bm = M[i,1]
                
                    # calculate deflections
                    alphaR     = (1/(1-m**2)) * Rs * (Am * np.cos(m*imTheta) + Bm * np.sin(m*imTheta))
                    alphaTheta = (m/(1-m**2)) * Rs * (-Am * np.sin(m*imTheta) + Bm * np.cos(m*imTheta))
                
                    # convert back to cartesian coords
                    alphaX = alphaR * np.cos(imTheta) + alphaTheta * (-np.sin(imTheta))
                    alphaY = alphaR * np.sin(imTheta) + alphaTheta * (np.cos(imTheta))
                
                    # add shear to current deflection angles, in arcsec
                    self.alpha_x += alphaX * 3600 * 180 / np.pi
                    self.alpha_y += alphaY * 3600 * 180 / np.pi
        
        delattr(self,'Multipoles')
        
        return
        
# ----------------------------------------------------------------------