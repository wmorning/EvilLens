"""
An object class to exist as part of a gravitational lens system
should be able to load itself from a file, know its own redshift
know its own x and y grid coordinates, and be able to interpret
between their pixel size and physical scale in arcsec.
"""
# ======================================================================

import numpy as np
from astropy.io import fits 

# ======================================================================

class Source(object):
    
    def __init__(self, Zs):  
    
        self.Zs = Zs
    
        return    
# ----------------------------------------------------------------------
    
    def Read_Source_from(self, fitsfile):
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
              
        
        assert len(self.intensity.shape) == 2
        if self.intensity.shape == (self.hdr['NAXIS1'],self.hdr['NAXIS2']):
            self.NX,self.NY = self.intensity.shape
        elif self.intensity.shape ==(self.hdr['NAXIS2'],self.hdr['NAXIS1']):
            self.NY,self.NX = self.intensity.shape
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
 
        
        
# ======================================================================
