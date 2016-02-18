# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 22:53:20 2014

@author: warrenmorningstar
"""
# ======================================================================

from astropy import units, constants
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
        self.q3 = self.q
        self.centroid = [0.01,0.01]  #WRM: centroid offsets center of map from
        self.r_c = 0.0               #     origin to avoid divergence at a pixel
        self.inclination=0.0
        self.rotation = 0.0
        return    
        
# ----------------------------------------------------------------------
 
    def build_kappa_map(self, sigma=None,q=None,centroid=None, r_c = None,rotation=0):
        '''
        Create kappa map using current parameters.  Also allows 
        user to manually set lens parameters if they want to.
        - sigma is central stellar velocity dispersion in km/s
        - q3 is axis ratio of oblate ellipsoid
        - centroid is x,y coordinates of center
        - r_c is core radius, in arcseconds
        - inclination is the inclination relative to the lens plane
        - rotation is the angle between the semi-major axis and the x-axis 
        '''
        if sigma is not None:
            self.sigma = sigma * units.km/units.s
        if q is not None:
            self.q = q
        if centroid is not None:
            assert len(centroid)==2
            self.centroid = centroid
        if r_c is not None:
            self.r_c = r_c
        if rotation is not None:
            self.rotation = rotation
        
        #rotate axes by the rotation angle around the centroid of the lens
        xprime = np.cos(self.rotation)*(self.x-self.centroid[0])+np.sin(self.rotation)*(self.y-self.centroid[1])
        yprime = -np.sin(self.rotation)*(self.x-self.centroid[0])+np.cos(self.rotation)*(self.y-self.centroid[1])
         
        #compute Einstein radius, in arcsec:
        b = 4*np.pi *((self.sigma/constants.c)**2)*self.Dds/self.Ds
        self.b = b.decompose()*(3600.0*180.0/np.pi)
         
         # Now compute kappa using lens equation from Saas Fe.
        #self.kappa = self.b.value /(2.0*np.sqrt(self.q**2*((self.x-self.centroid[0])**2+self.r_c**2)+(self.y-self.centroid[1])**2))
        
        #compute kappa in rotated frame.        
        #self.kappa = self.b.value /(2.0*np.sqrt(self.q**2*((xprime)**2+self.r_c**2)+(yprime)**2))
        
        
        #compute kappa in rotated frome using equation from Koorman, Scheinder
        # & Bartelmann 1994
#       1.   convert xprime,yprime to radians, and get r,theta
        xprime *=np.pi/3600.0/180.0
        yprime *=np.pi/3600.0/180.0
        r = np.sqrt(xprime**2+self.q**2*yprime**2)/b.decompose().value
        bc = self.r_c/3600.0/180.0*np.pi/b.decompose().value
        self.kappa = np.sqrt(self.q)/(2.0*np.sqrt(r**2+bc**2))
        
#       2.           
        
        return
         
# ----------------------------------------------------------------------
        
    def deflect(self):
        
        # First convert image coordinates to angle from semimajor axis.
        #image_theta = np.arctan2(self.image_y-self.centroid[1],self.image_x-self.centroid[0]) # check...
        
        #rotate axes by the rotation angle around the centroid of the lens
        xprime = np.cos(self.rotation)*(self.image_x-self.centroid[0])+np.sin(self.rotation)*(self.image_y-self.centroid[1])
        yprime = -np.sin(self.rotation)*(self.image_x-self.centroid[0])+np.cos(self.rotation)*(self.image_y-self.centroid[1])
        
        
        # Deflect analytically.  This is old way, but it should be the same.
        #self.alpha_x = (self.b.value/np.sqrt(1-self.q**2))*np.arctan(np.sqrt((1-self.q**2)/(self.q**2*np.cos(image_theta)**2+np.sin(image_theta)**2))*np.cos(image_theta))
        #self.alpha_y = (self.b.value/np.sqrt(1-self.q**2))*np.arctanh(np.sqrt((1-self.q**2)/(self.q**2*np.cos(image_theta)**2+np.sin(image_theta)**2))*np.sin(image_theta))
        
        # Deflect analytically using equation from Saas Fe
        #self.alpha_x = (self.b.value/np.sqrt(1-self.q**2)) * np.arctan((self.image_x-self.centroid[0])*np.sqrt(1-self.q**2)/(np.sqrt(self.q**2*((self.image_x-self.centroid[0])**2+self.r_c**2)+(self.image_y-self.centroid[1])**2)+self.r_c))        
        #self.alpha_y = (self.b.value/np.sqrt(1-self.q**2)) * np.arctanh((self.image_y-self.centroid[1])*np.sqrt(1-self.q**2)/(np.sqrt(self.q**2*((self.image_x-self.centroid[0])**2+self.r_c**2)+(self.image_y-self.centroid[1])**2)+self.q**2*self.r_c))
        
        
        
        #Deflect analytically for rotated SIE.  Then rotate back to original coordinate bases.

        #Do this using eq 27a from Koorman, Schneider, & Bartelmann
        phi = np.arctan2(yprime,xprime)
        self.phi = phi
        qprime = np.sqrt(1-self.q**2 )     
        alpha_x_prime = self.b.value*(np.sqrt(self.q)/qprime)*(np.arcsinh(np.cos(phi)*qprime/self.q))
        alpha_y_prime = self.b.value*(np.sqrt(self.q)/qprime)*np.arcsin(qprime*np.sin(phi))

        self.alpha_x = np.cos(self.rotation)*alpha_x_prime-np.sin(self.rotation)*alpha_y_prime
        self.alpha_y = np.sin(self.rotation)*alpha_x_prime+np.cos(self.rotation)*alpha_y_prime
         
        self.alpha_x_prime = alpha_x_prime
        self.alpha_y_prime = alpha_y_prime
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
    def get_mass_inside(self,r):
        '''
        Returns mass inside a given radius r, where r is in kpc
        '''
        r = r*units.kpc
        mass = (np.pi*self.sigma**2/constants.G*r).to(units.solMass)
        
        return mass

# ----------------------------------------------------------------------
    def print_mass_inside(self, r):
        '''
        Print mass inside a given radius r, where r is in kpc
        '''
        Mass_encl = self.get_mass_inside(r)
        print('Mass Enclosed in ',r,' kpc is ',Mass_encl,' solar masses \n')

# ======================================================================

if __name__ == '__main__':

    SIElens = evil.AnalyticSIELens(0.4,1.5, q=0.75, centroid=[.01,.01])
    
    SIElens.deflect()
    print SIElens.alpha_x
    
    SIElens.build_kappa_map()
    
        
# ======================================================================
