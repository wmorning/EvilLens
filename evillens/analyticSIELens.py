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
        
        # Compute kappa in rotated frome using equation from Koorman, Scheinder
        # & Bartelmann 1994
        xprime *=np.pi/3600.0/180.0
        yprime *=np.pi/3600.0/180.0
        r = np.sqrt(xprime**2+self.q**2*yprime**2)/b.decompose().value
        bc = self.r_c/3600.0/180.0*np.pi/b.decompose().value
        self.kappa = np.sqrt(self.q)/(2.0*np.sqrt(r**2+bc**2))
        
        return
         
# ----------------------------------------------------------------------
        
    def deflect(self):
        '''
        Analytically calculate deflection angles at all the image pixel
        positions.  No argunents required as long as lens has been 
        constructed.  Returns assertion error if it hasn't.
        '''
        assert self.kappa is not None
        
        #rotate axes by the rotation angle around the centroid of the lens
        xprime = np.cos(self.rotation)*(self.image_x-self.centroid[0])+np.sin(self.rotation)*(self.image_y-self.centroid[1])
        yprime = -np.sin(self.rotation)*(self.image_x-self.centroid[0])+np.cos(self.rotation)*(self.image_y-self.centroid[1])
        
        #Deflect analytically for rotated SIE.  Then rotate back to original coordinate bases.
        #Do this using eq 27a from Koorman, Schneider, & Bartelmann
        phi = np.arctan2(yprime,xprime)
        self.phi = phi
        if np.isclose(self.q,1.): # Avoid NaN for q=1 using SIS.
            alpha_x_prime = self.b.value*np.cos(phi)
            alpha_y_prime = self.b.value*np.sin(phi)         
        else:
            if np.isclose(self.r_c,0):
                raise Exception("cannot include core radius yet\n")
                x1 = xprime 
                x2 = yprime
                x = np.sqrt(xprime**2+self.q**2*yprime**2)/self.b.value
                bc = self.r_c
                qp = np.sqrt(1-self.q**2)
                Qp = ((qp*np.sqrt(self.b**2+bc**2)+x1)**2 +self.q**4 *x2**2) \
                    / ((self.q*x+qp*bc*x1)**2+qp**2 * bc**2 *x2**2)
                Qm = ((qp*np.sqrt(self.b**2+bc**2)-x1)**2+self.q**4*x2**2) \
                    / ((self.q*x-qp*bc*x1)**2+qp**2 * bc**2 *x2**2)
                
                R  = (x1**2 +self.q**4*x2**2 - qp**2 * (self.b**2+bc**2) - 2j*self.q**2*qp*np.sqrt(self.b**2+bc**2)*x2).value
                S  = self.q**2*x**2 - qp**2 * bc**2 - 2j*self.q*qp*bc*x2
                
                print R
                print S
                
                self.R = R
                self.S = S
                
                alpha_x_prime = np.sqrt(self.q)/(4*qp) * np.log(Qp/Qm)
                alpha_y_prime = -np.sqrt(self.q)/(2*qp)  * (np.angle(R)-np.angle(S))
                
                
                
            else:
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
        imR     = np.sqrt((self.image_x-self.centroid[0])**2+(self.image_y-self.centroid[1])**2) 
        imTheta = np.arctan2(self.image_y-self.centroid[1],self.image_x-self.centroid[0])
        
        for i in range(Nmoments):
            m = i+2  # The first multipole is m=2 (shear)
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
                self.alpha_x -= alphaX 
                self.alpha_y -= alphaY 
            else:  # higher order multipoles
                Am = M[i,0] 
                Bm = M[i,1]
                
                # calculate deflections
                alphaR     = (1/(1-m**2)) * (Am * np.cos(m*imTheta) + Bm * np.sin(m*imTheta))
                alphaTheta = (m/(1-m**2)) * (-Am * np.sin(m*imTheta) + Bm * np.cos(m*imTheta))
                
                # convert back to cartesian coords
                alphaX = alphaR * np.cos(imTheta) + alphaTheta * (-np.sin(imTheta))
                alphaY = alphaR * np.sin(imTheta) + alphaTheta * (np.cos(imTheta))
                
                # add shear to current deflection angles
                self.alpha_x -= alphaX 
                self.alpha_y -= alphaY 
            
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
