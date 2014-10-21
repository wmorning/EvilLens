# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 22:53:20 2014

@author: warrenmorningstar
"""
# ======================================================================

from astropy import units, constants
from math import pi
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================

class GravitationalLens(object):
    '''
    An object class describing a gravitational lens system.
    '''
    def __init__(self, Zd, Zs):
        
        self.Zd = Zd
        self.Zs = Zs

        # Calculate distances and the critical density:
        self.cosmological = FlatLambdaCDM(H0=70, Om0=0.3)
        self.compute_distances()
        
        # Make a default pixel grid:
        self.setup_grid(NX=100,NY=100,pixscale=0.1, n=1,offset=0.5)
        
        return

# ----------------------------------------------------------------------
        
    def compute_distances(self):

        Dd = self.cosmological.angular_diameter_distance(self.Zd)
        Ds = self.cosmological.angular_diameter_distance(self.Zs)
        Dds = self.cosmological.angular_diameter_distance_z1z2(self.Zd,self.Zs)
        SigmaCrit = constants.c**2 /(4*pi*constants.G) * Ds/(Dd*Dds)
        
        self.Dd = Dd
        self.Ds = Ds 
        self.Dds = Dds 
        self.SigmaCrit = units.Quantity.to(SigmaCrit,units.solMass/units.Mpc**2)
        
        return
 
# ----------------------------------------------------------------------

    def setup_grid(self,NX=None,NY=None,pixscale=None, n=None , offset=None):
        '''
        Make two arrays, x and y, that define the extent of the maps
        - pixscale is the size of a pixel, in arcsec.
        - n is oversampling factor between kappa and image maps
        - offset is diagonal offset of kappa and image pixels
        '''
        if NX is not None: 
            self.NX = NX
        if NY is not None: 
            self.NY = NY
        if pixscale is not None: 
            self.pixscale = pixscale
        if n is not None:
            self.n = n
        if offset is not None:
            self.offset = offset
            
        xgrid = np.arange(-self.NX/2.0,(self.NX)/2.0,1.0)*self.pixscale+self.pixscale
        ygrid = np.arange(-self.NY/2.0,(self.NY)/2.0,1.0)*self.pixscale+self.pixscale
        
        #WRM:  here we build new grid for the image and source pixels,
        #      purposefully misaligned with the kappa pixels, so no NaNs occur.        
        self.pixel_offset = self.offset*self.pixscale/self.n
        image_xgrid = np.arange(-self.NX/2.0,(self.NX)/2.0,1.0/self.n)*self.pixscale-self.pixel_offset
        image_ygrid = np.arange(-self.NY/2.0,(self.NY)/2.0,1.0/self.n)*self.pixscale-self.pixel_offset
        
        self.x, self.y = np.meshgrid(xgrid,ygrid)
        self.image_x, self.image_y = np.meshgrid(image_xgrid,image_ygrid)
        self.NX_image,self.NY_image = self.image_x.shape
        return
        
# ----------------------------------------------------------------------
 
    def build_kappa_map(self):
        self.kappa = None
        return
        
# ----------------------------------------------------------------------
       
    def read_kappa_from(self,fitsfile):
        '''
        Read a convergence map from a FITS format file, and adopt its
        pixel grid. This means we have to extract the pixel scale from 
        the FITS header.
        '''
        if fitsfile is None:
            raise Exception("No kappa map FITS image provided.\n") 
        
        # Open the file and read the image data:
        hdulist = fits.open(fitsfile)
        self.hdr = hdulist[0].header
        self.kappa = hdulist[0].data
        hdulist.close()
        
        # Extract the pixel grid information:
        assert len(self.kappa.shape) == 2
        assert self.kappa.shape == (self.hdr['NAXIS1'],self.hdr['NAXIS2'])
        self.NX,self.NY = self.kappa.shape
        self.set_pixscale()
        
        # Set up a new pixel grid to go with this new kappa map:
        self.setup_grid()
        
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
    
    def deflect(self):
        
        if self.kappa is None:
            self.alpha_x = None
            self.alpha_y = None  
        elif len(self.kappa.shape) == 2:
            
            #create empty arrays to be filled with x,y components of alpha
            alpha_x = np.empty([self.NX_image,self.NY_image], float)
            alpha_y = np.empty([self.NX_image,self.NY_image], float)
            
            #double for loop to get each point in array
            for i in range(len(alpha_x[:,0])):
                for j in range(len(alpha_x[0,:])):
                    alpha_x[i,j] =1/np.pi * np.nansum(self.kappa * (self.image_x[i,j]-self.x)/((self.image_x[i,j]-self.x)**2+(self.image_y[i,j]-self.y)**2)*self.pixscale**2)
                    alpha_y[i,j] =1/ np.pi * np.nansum(self.kappa * (self.image_y[i,j]-self.y)/((self.image_x[i,j]-self.x)**2+(self.image_y[i,j]-self.y)**2)*self.pixscale**2)
            self.alpha_x = alpha_x
            self.alpha_y = alpha_y
        else:
            raise Exception("Can't do integral.  your kappa map must be 2-D .\n")  
        
        return
    
# ----------------------------------------------------------------------    
    
    def plot(self,mapname):    
        '''
        Plot the given map as a nice colorscale image, with contours.
        '''
        # Which map do we want to plot?
        # And what options does that mean we need?
        if mapname == "kappa":
            img = self.kappa
            levels = np.arange(0.1,1.5,0.2)
            options = dict(interpolation='nearest',\
                           origin='lower',\
                           vmin=-0.2, \
                           vmax=1.5)       
                           
        elif mapname == "alpha_x":
            img = self.alpha_x
            levels = np.arange(-0.5,0.5,0.1)
            options = dict(interpolation='nearest',\
                           origin='lower',\
                           vmin=-0.5, \
                           vmax=0.5)
        elif mapname == "alpha_y":
            img = self.alpha_y
            levels = np.arange(-0.5,0.5,0.1)
            options = dict(interpolation='nearest',\
                           origin='lower',\
                           vmin=-0.5, \
                           vmax=0.5)
        else:
             raise ValueError("unrecognized map name %s" % mapname)
        
        # Generic figure setup:
        px,py = 1,1
        figprops = dict(figsize=(5*px,5*py), dpi=128)
        adjustprops = dict(left=0.1,\
                           bottom=0.1,\
                           right=0.95,\
                           top=0.95,\
                           wspace=0.1,\
                           hspace=0.1)
        
        # Finish setting up the options.
        # 1) Images need extents, if x and y are not pixel numbers:
        options['extent'] = (np.min(self.x),np.max(self.x),\
                                 np.min(self.y),np.max(self.y))
        # 2) The cubehelix map is linear grayscale on a BW printer
        options['cmap'] = plt.get_cmap('cubehelix')
        
        # Start the figure:
        fig = plt.figure(**figprops)
        fig.subplots_adjust(**adjustprops)
        plt.clf()

        # Plot a colored pixel map and overlay some contours:
        plt.imshow(img, **options)
        
        
        # WRM: Only plot contour map for kappa.  Skip for alpha.
        if mapname == "kappa":
            plt.contour(self.x, self.y, img, levels,colors=('k',))
        else:
            pass
        
        
        
        # Annotate the plot:
        plt.xlabel('x / arcsec')
        plt.ylabel('y / arcsec')
        plt.axes().set_aspect('equal')

        # If we're in a notebook, display the plot. 
        # Otherwise, make a PNG.
        try:
            __IPYTHON__
            plt.show()
        except NameError:
            pngfile = mapname+'.png'
            plt.savefig(pngfile)
            print "Saved plot to "+pngfile
            
        return


# ----------------------------------------------------------------------

    def write_kappa_to( self, fitsfile="kappa_map.fits"):
        
        hdu = fits.PrimaryHDU(self.kappa)
        
        hdu.header['CDELT1'] = self.pixscale / 3600.0
        hdu.header['CDELT2'] = self.pixscale / 3600.0
        
        hdu.writeto(fitsfile)
       
        return
        
# ----------------------------------------------------------------------

    def raytrace(source_image):
        raise Exception("Can't do raytracing yet.\n")  
        return

# ======================================================================

if __name__ == '__main__':

    lens = GravitationalLens(0.4,1.5)
    
    print "Difference in angular diameter distances: ",lens.Ds - lens.Dd
    print "  cf. Dds = ", lens.Dds
    print "Critical density = ",lens.SigmaCrit

    lens.read_kappa_from("examples/test_kappa.fits")
    lens.plot("kappa")
            
# ======================================================================
