import numpy as np
from astropy import units,constants
import struct
from scipy.interpolate import interp1d
import scipy.special as sp

def Sersic(x,y,x0,y0,q,r_eff,phi,n,bn):
    
    # rotate x,y axes and shift relative to center
    xp = np.cos(phi)*(x-x0)-np.sin(phi)*(y-y0)
    yp = np.sin(phi)*(x-x0)+np.cos(phi)*(y-y0)
    
    # compute scaled (half-light) radius and intensity
    rscl = np.sqrt(q*xp**2+yp**2/q)/r_eff
    I = np.exp(-bn * (rscl**(1./n)-1))
    
    return I
    

    
def Compute_bn(n,error=10**-8):
    def fx(n,bn):
        return(2*sp.gammainc(2*n,bn)-1)
    
    #initial guess follows approximation from Capaccioli 1989
    x0 = 1.9992*n - 0.3271
    x1 = x0 + 0.01
    j = 0
    epsilon = abs(x1-x0)
    while epsilon > error and j<100000:
        fx0 = fx(n,x0)
        fx1 = fx(n,x1)
        x0,x1 = x1 , x1 - fx1*(x1-x0) / (fx1-fx0)
        epsilon = abs(x1-x0)
        j+=1
    if j ==100000:
        #solution didn't converge, lets just approximate it.
        bn = 1.9992*n-0.3271
        print 'solution didn"t converge'
    else:
        bn = x1
        
    return bn
    
def Subhalo_cumulative_mass_function(subhalo_mass,halo_mass):
    mu = subhalo_mass / halo_mass
    
    mu_til = 0.01
    mu_cut = 0.096
    a = -0.935
    b = 1.29 
    
    return((mu/mu_til)**a * np.exp(- (mu / mu_cut)**b ) ) 
    

def Subhalo_Mass_function(subhalo_mass,halo_mass):
    
    mu = subhalo_mass / halo_mass
    
    mu_til = 0.01
    mu_cut = 0.096
    a = -0.935
    b = 1.29 
    
    return((b*(mu/mu_cut)**b -a) * Subhalo_cumulative_mass_function(subhalo_mass,halo_mass))

def Einasto(r,alpha,scale):
    return(np.exp(-(2/alpha)*((r/scale)**alpha-1.))*scale)