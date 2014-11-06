"""
Image converter script for Jpg files to make fits
"""


import Image

from astropy.io import fits as fits
import numpy as np
import matplotlib.pyplot as plt


def JPG_to_FITS(jpgimage):

    image = Image.open(jpgimage)
    image.getcolors()
    r,g,b = image.split()
    
    xsize,ysize = image.size  
    
    rdata = r.getdata()    
    gdata = g.getdata()
    bdata = b.getdata()
    
    npr = np.reshape(rdata, (ysize, xsize))
    npg = np.reshape(gdata, (ysize, xsize))
    npb = np.reshape(bdata, (ysize, xsize))
    
    magnitude = np.sqrt(npr**2+npg**2+npb**2)
    
    hdu = fits.PrimaryHDU(data=magnitude)
    hdu.header['CDELT1'] = 9.0/ysize/3600.0
    hdu.header['CDELT2'] = 9.0/ysize/3600.0
    hdu.header['NAXIS1'],hdu.header['NAXIS2'] = magnitude.shape
    print(hdu.header)
    hdu.writeto('output.fits', clobber=True)
    
    print(hdu.header)
    print(hdu.filename())
    
    return 
    
