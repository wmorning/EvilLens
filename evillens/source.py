"""
An object class to exist as part of a gravitational lens system
should be able to load itself from a file, know its own redshift
know its own x and y grid coordinates, and be able to interpret
between their pixel size and physical scale in arcsec.
"""
# ======================================================================

import numpy as np
from astropy.io import fits 
from astropy import units,constants
from astropy.cosmology import FlatLambdaCDM
import scipy.special as sp
import math

# ======================================================================

class Source(object):
    
    def __init__(self, Zs):  
    
        self.Zs = Zs
        self.compute_distances()
    
        return    
# ----------------------------------------------------------------------
    def compute_distances(self):
        self.cosmological = FlatLambdaCDM(H0=71.0, Om0=0.2669)
        self.Ds = self.cosmological.angular_diameter_distance(self.Zs)
# ----------------------------------------------------------------------   
    def read_source_from(self, fitsfile):
        '''
        Read an image from a fitsfile, and setup its grid.  Here we need
        to properly read in the wcs coordinate information so that the 
        grid is spaced correctly. This means we have to extract the pixel 
        scale from the FITS header.
        '''
        if fitsfile is None:
            raise Exception("You need to provide an image.\n")
            
        hdulist = fits.open(fitsfile)
        self.hdr = hdulist[0].header
        self.intensity = hdulist[0].data
        hdulist.close()
              
        if self.hdr['NAXIS'] == 2:
            if self.intensity.shape == (self.hdr['NAXIS1'],self.hdr['NAXIS2']):
                self.NX,self.NY = self.intensity.shape
            elif self.intensity.shape ==(self.hdr['NAXIS2'],self.hdr['NAXIS1']):
                self.NY,self.NX = self.intensity.shape
            else:
                raise Exception("Your image is formatted incorrectly.\n")    
        else:
            assert len(self.intensity.shape) == 3
            if self.intensity.shape == (self.hdr['NAXIS'],self.hdr['NAXIS1'],self.hdr['NAXIS2']):
                self.Naxes,self.NX,self.NY = self.intensity.shape
            elif self.intensity.shape ==(self.hdr['NAXIS'],self.hdr['NAXIS2'],self.hdr['NAXIS1']):
                self.Naxes,self.NY,self.NX = self.intensity.shape
            else:
                raise Exception("Your image is formatted incorrectly.\n")
        self.set_pixscale()
        
        # Set up a new pixel grid to go with this new kappa map:
        self.setup_grid()
        
        return
# ----------------------------------------------------------------------
    
    def setup_grid(self, NX=None, NY=None, pixscale=None):
        '''
        Make two arrays, x and y, that define the extent of the maps
        - pixscale is the size of a pixel, in arcsec.
        - 
        '''
        if NX is not None: 
            self.NX = NX
        if NY is not None: 
            self.NY = NY
        if pixscale is not None: 
            self.pixscale = pixscale        
        
        xgrid = np.arange(-self.NX/2.0,(self.NX)/2.0,1.0)*self.pixscale+self.pixscale
        ygrid = np.arange(-self.NY/2.0,(self.NY)/2.0,1.0)*self.pixscale+self.pixscale
        
        self.beta_x, self.beta_y = np.meshgrid(xgrid,ygrid)        
        
        return

# ----------------------------------------------------------------------
    
    def set_pixscale(self):
        
        # Modern FITS files:
        if 'CD1_1' in self.hdr.keys():            
            determinant = self.hdr['CD1_1']*self.hdr['CD2_2'] \
                          - self.hdr['CD1_2']*self.hdr['CD2_1']
            self.pixscale = 3600.0*np.sqrt(np.abs(determinant))

        # Older FITS files:
        elif 'CDELT1' in self.hdr.keys():
            self.pixscale = 3600.0*np.sqrt(np.abs(self.hdr['CDELT1']*self.hdr['CDELT2']))

        # Simple FITS files with no WCS information (bad):
        else:
            self.pixscale = 1.0
            
        return        

# ----------------------------------------------------------------------
    def build_from_clumps(self,size=2.0,clump_size = 0.1,axis_ratio=1.0, orientation=0.0,center=[0,0], Nclumps=50, n = 1 , error =10**-8,singlesource=False,seeds=[1,2,3],Flux=1.0):
        #raise Exception("cannot build source from clumps yet. \n")
        '''
        Build source from gaussian clumps centered about specified position.
        Accepted parameters are as follows:
        - Size of the source, in kpc (approximately the half light radius)
        - Size of individual clumps.  RMS determines spread in clump size.
        - Axis ratio, or the ratio of the major axis to the minor axis.
        - Orientation angle of the source measured in radians from the 
           positive x-axis.
        - Position of the source center (in arcsec).
        - The number of clumps making up the source.
        - The average brightness of each clump.
        - The standard deviation of the brightness for individual clumps.
           The clump brightness follows a lognormal distribution
        - Sersic index n
        '''
        
        self.axis_ratio = axis_ratio
        self.orientation = orientation
        self.Nclumps = Nclumps
        self.center = center
        self.n = n
        self.Flux=Flux
        #compute rms radius in arcsec
        self.size = np.arctan((size *units.kpc) / self.Ds ).to(units.arcsec).value
        self.clump_size = np.arctan((clump_size*units.kpc)/self.Ds).to(units.arcsec).value 
        #pick random positions inside of 4r_eff          
#        rlist = np.sqrt(np.random.random(Nclumps))*4.0*self.size
#        thetalist = np.random.random(Nclumps)*2*np.pi
        if singlesource ==False:
            # seed random # generation
            np.random.seed(seeds[0])
            xpos_orig = np.random.exponential(self.size,Nclumps)/np.sqrt(self.axis_ratio)*np.random.choice([-1,1],Nclumps)
            np.random.seed(seeds[1])
            ypos_orig = np.random.exponential(self.size,Nclumps)*np.sqrt(self.axis_ratio)*np.random.choice([-1,1],Nclumps)
            np.random.seed(seeds[2])
            self.xlist = xpos_orig*np.cos(self.orientation)-ypos_orig*np.sin(self.orientation) + self.center[0]
            self.ylist = xpos_orig*np.sin(self.orientation)+ypos_orig*np.cos(self.orientation) +self.center[1]
        else:
            self.xlist = np.array([center[0],100000000])
            self.ylist = np.array([center[1],100000000])   
        #determine constant b_n which allows us to use r as half light radius
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
            self.b_n = 1.9992*n-0.3271
        else:
            self.b_n = x1
        
        #self.Blist = np.exp(-self.b_n*((np.sqrt((np.cos(self.orientation)*(self.xlist-self.center[0])-np.sin(self.orientation)*(self.ylist-self.center[1]))**2*self.axis_ratio+((self.xlist-self.center[0])*np.sin(self.orientation)+(self.ylist-self.center[1])*np.cos(self.orientation))**2/self.axis_ratio)/self.size)**(1/self.n)-1))   
        np.random.seed(seeds[2])     
        self.Slist = np.random.exponential(self.clump_size,self.Nclumps)
        
        for i in range(self.Nclumps):
            if i==0:
                self.intensity = (1.0/np.sqrt(2*np.pi*self.Slist[i]**2))*np.exp(-0.5*((self.beta_x-self.xlist[i])**2+(self.beta_y-self.ylist[i])**2)/self.Slist[i]**2)
            else:
                self.intensity +=(1.0/np.sqrt(2*np.pi*self.Slist[i]**2))*np.exp(-0.5*((self.beta_x-self.xlist[i])**2+(self.beta_y-self.ylist[i])**2)/self.Slist[i]**2)
        

        self.intensity *= np.exp(-self.b_n*((np.sqrt((np.cos(self.orientation)*(self.beta_x-self.center[0])+np.sin(self.orientation)*(self.beta_y-self.center[1]))**2*self.axis_ratio+(-(self.beta_x-self.center[0])*np.sin(self.orientation)+(self.beta_y-self.center[1])*np.cos(self.orientation))**2/self.axis_ratio)/self.size)**(1/self.n)-1))        
        
        # Normalize flux of all sources to input total flux
        self.intensity *= self.Flux/(np.sum(self.intensity)*self.pixscale**2)
        return
        
# ----------------------------------------------------------------------

    def write_source_to(self,fitsfile,overwrite=False):
        '''
        Write an image of the source to a fits file.
        '''
        
        hdu = fits.PrimaryHDU(self.intensity)
        hdu.header['CDELT1'] = self.pixscale / 3600.0
        hdu.header['CDELT2'] = self.pixscale / 3600.0
        hdu.writeto(fitsfile, clobber=overwrite)
        
        return
        
# ======================================================================
