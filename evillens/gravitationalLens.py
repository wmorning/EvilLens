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
from scipy import interpolate, spatial
from scipy.integrate import simps
from time import time

# ======================================================================

class GravitationalLens(object):
    '''
    An object class describing a gravitational lens system.
    '''
    def __init__(self, Zd, Zs):
        
        self.Zd = Zd
        self.Zs = Zs
        self.source = None
        self.alpha_x = None
        self.alpha_y = None
        self.kappa = None

        # Calculate distances and the critical density:
        self.cosmological = FlatLambdaCDM(H0=71.0, Om0=0.2669)
        self.compute_distances()
        
        # Make a default pixel grid:
        self.setup_grid(NX=100,NY=100,pixscale=0.1, n=1, n2=1,offset=0.5)
        
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

    def setup_grid(self,NX=None,NY=None,pixscale=None, n=None , n2=None, offset=None):
        '''
        Make two arrays, x and y, that define the extent of the maps
        - pixscale is the size of a pixel, in arcsec.
        - n is oversampling factor between kappa and image maps
        - n2 is size of image pixels relative to density pixels
        - offset is diagonal offset of kappa and image pixels
        '''
        if NX is not None: 
            self.NX = NX
        if NY is not None: 
            self.NY = NY
        if pixscale is not None: 
            self.pixscale = pixscale
        if n is not None:
            self.n = int(n)
        if n2 is not None:
            self.n2 = int(n2)
        if offset is not None:
            self.offset = offset
        
        
        #build grid for kappa map
        self.xgrid = np.arange(-self.NX/2.0,(self.NX)/2.0,1.0)*self.pixscale+self.pixscale
        self.ygrid = np.arange(-self.NY/2.0,(self.NY)/2.0,1.0)*self.pixscale+self.pixscale
        self.x, self.y = np.meshgrid(self.xgrid,self.ygrid)        
        
        
        #WRM:  here we build new grid for the image and source pixels,
        #      purposefully misaligned with the kappa pixels, so no NaNs occur.        
        self.pixel_offset = self.offset*self.pixscale
        image_xgrid = np.arange(-(self.NX//self.n)/2.0,(self.NX//self.n)/2.0,1.0)*self.n2*self.pixscale+self.n2*self.pixscale-self.pixel_offset
        image_ygrid = np.arange(-(self.NY//self.n)/2.0,(self.NY/self.n)/2.0,1.0)*self.n2*self.pixscale+self.n2*self.pixscale-self.pixel_offset
        self.image_x, self.image_y = np.meshgrid(image_xgrid,image_ygrid)
        self.NX_image,self.NY_image = self.image_x.shape        
        
        
        
        
        return
        
# ----------------------------------------------------------------------
 
    def build_kappa_map(self, q, M, gamma, centroid=[0.025,0.025],rotation=0.0):
        '''
        Create kappa map using elliptical power-law profile.
        M is Einstein mass (in solar masses), q is axis ratio, and gamma is
        power-law index
        '''
        self.q = q
        self.gamma = gamma
        self.centroid = centroid
        ThetaE = (4.0*constants.G*(M* units.solMass)/constants.c**2 *self.Dds/(self.Dd*self.Ds))**(1.0/2.0)
        self.ThetaE = ThetaE.decompose()*3600.0*180.0/np.pi
        self.rotation = rotation        
        
        assert self.x.shape == self.y.shape
        
        xprime = np.cos(self.rotation)*(self.x-self.centroid[0])+np.sin(self.rotation)*(self.y-self.centroid[1])
        yprime = -np.sin(self.rotation)*(self.x-self.centroid[0])+np.cos(self.rotation)*(self.y-self.centroid[1])
            
        
        
        self.kappa = (3.0-gamma)/2.0 *(self.ThetaE.value/np.sqrt(self.q*xprime**2+yprime**2/self.q))**(gamma-1)
        return
        
    def build_kappa_from(self,NbodySim,Np,mpart,bext,NX,NY,pixscale,n=1,n2=1,method='TCM'):
        '''
        Read the outputs of an N-body simulation, right now requires simulation parameters to be input, but
        this could in principle be self determined.
        '''
        
        raise Exception('Cannot build from N-body simulation yet. \n')
        
        p = np.loadtxt( NbodySim )             # get variables p and ids, probably not this way
        ids = np.loadtxt(ids).reshape([Np,Np,Np])
        
        # The following 20 lines are part of a notebook sent to me by Tom Abel that are
        # used in the pre-processing of his simulations.  I'm not really sure what it is
        # doing however, and it may not be a necessary step.
        Ndim = Np**(0.33333333334)
        x = bext[0]*(0.5+np.arange(Ndim))/(Ndim)
        xg = np.meshgrid(x,x,x, indexing='ij')
        xl = np.ravel(xg[0])
        yl = np.ravel(xg[1])
        zl = np.ravel(xg[2])
        
        dx = (p[:,0]-xl)
        dy = (p[:,1]-yl)
        dz = (p[:,2]-zl)
        dis = np.sqrt(dx**2+dy**2+dz**2) 
        (p[:,0])[dx < -bext[0]/2.0] += bext[0] # shift particles that arent in the box.
        (p[:,1])[dy < -bext[1]/2.0] += bext[1]
        (p[:,2])[dz < -bext[2]/2.0] += bext[2]
        (p[:,0])[dx >  bext[0]/2.0] -= bext[0]
        (p[:,1])[dy >  bext[1]/2.0] -= bext[1]
        (p[:,2])[dz >  bext[2]/2.0] -= bext[2]
        dx = (p[:,0]-xl)
        dy = (p[:,1]-yl)
        dz = (p[:,2]-zl)
        dis = np.sqrt(dx**2+dy**2+dz**2)
        
        # Define the Tetrahedra and their vertices for each cubic cell
        vert = np.array(((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)) )
        conn = np.array(((1,0,2,4), (3,1,2,4), (3,5,1,4), (3,6,5,4), (3,2,6,4), (3,7,5,6)))
        
        # Build the Tetrahedra
        Ntetpp = len(conn)
        Ntet = (Ndim-1)**3*Ntetpp
        tet = np.zeros((Ntet,4),dtype=int)     # Tetrahedron member IDs
        cen = np.zeros((Ntet,3))               # Tetrahedron center positions
        m = np.zeros(Ntet,dtype=float)         # Tetrahedron masses (starts as volumes)
        
        for m in xrange(Ntetpp):
            for n in xrange(4):
                ip = vert[conn[m]][n,0]        # offset in the i dimension
                jp = vert[conn[m]][n,1]        # offset in the j dimension
                kp = vert[conn[m]][n,2]        # offset in the k dimension
                
                tet[m::Ntetpp,n] = idslist[ip:Ndim-1+ip,jp:Ndim-1+jp,kp:Ndim-1+kp].flatten()
        
        # Locate barycenters of Tetrahedra
        cen[:,:] = (p[tet[:,0],:]+p[tet[:,1],:]+p[tet[:,2],:]+p[tet[:,3],:])/4.0
        
        # Determine the Tetrahedron volumes
        a = p[tet[:,0],:]-p[tet[:,3],:]
        b = p[tet[:,1],:]-p[tet[:,3],:]
        c = p[tet[:,2],:]-p[tet[:,3],:]
        m[:] = mpart*np.fabs(np.sum(a*np.cross(b,c),axis=1)/6.0)
        
        # Clear space
        a=0
        b=0
        c=0
        
        m /= np.prod(bext)/Ndim**3             # Normalize the masses     
        
        if method == 'TCM':
            pass
        elif method == 'T4PM':
            a = np.zeros(4*xc.shape[0],3)
            a[::4,:] = np.sqrt(5)*xn[tesl.simplices[0],:]/5.0+(1-np.sqrt(5/5.0))*xc
            a[1::4,:] = np.sqrt(5)*xn[tesl.simplices[1],:]/5.0+(1-np.sqrt(5/5.0))*xc
            a[2::4,:] = np.sqrt(5)*xn[tesl.simplices[2],:]/5.0+(1-np.sqrt(5/5.0))*xc
            a[3::4,:] = np.sqrt(5)*xn[tesl.simplices[3],:]/5.0+(1-np.sqrt(5/5.0))*xc
            cen = a
        elif method =='RTCM':
            raise Exception('No Recursive TCM yet \n')
        else:
            print('No method selected, assuming TCM \n')
        
        # Create grid used to deposit density
        self.setup_grid(NX=NX,NY=NY,pixscale=pixscale)
        self.x *= (np.pi*self.Dd/3600.0/180.0).to(units.Mpc).value
        self.y *= (np.pi*self.Dd/3600.0/180.0).to(units.Mpc).value
        Sigma = np.zeros(self.x.shape)
        
        
        # Discard anything that falls outside of map 
        cen = cen[np.where((cen[:,0]>np.min(self.x)) & (cen[:,0]<np.max(self.x)) \
                         & (cen[:,1]>np.min(self.y)) & (cen[:,1]<np.max(self.y))),:]
        
        # CIC interpolation to density grid for Sigma
        i = self.x[0,:].searchsorted(a[:,0])-1
        j = self.y[:,0].searchsorted(a[:,1])-1
        dx= cen[:,0]-self.x[0,i] 
        dy= cen[:,1]-self.y[j,0]
        
        for k in xrange(0,len(i)):
            Sigma[i[k],j[k]]    += m*(1-dx[k])*(1-dy[k])/4.0
            Sigma[i[k]+1,j[k]]  += m * dx[k] * (1-dy[k])/4.0
            Sigma[i[k],j[k]+1]  += m * (1-dx[k]) * dy[k]/4.0
            Sigma[i[k]+1,j[k]+1]+= m * dx[ k ] * dy[ k ]/4.0
        
        Sigma *= units.solMass/units.Mpc**2
        self.kappa = (Sigma/self.SigmaCrit).decompose().value
        self.x /= np.pi*self.Dd/3600.0/180.0
        self.y /= np.pi*self.Dd/3600.0/180.0
        
        print('kappa built successfully')
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
        # Include padding if it was given in the fits header
        if 'NPADDING' in self.hdr.keys():
            self.n = self.hdr['NPADDING']
        else:
            self.n = 1
        if 'NPAD2' in self.hdr.keys():
            self.n2 = self.hdr['NPAD2']
        else:
            self.n2 = 1
            
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
    
    def deflect(self, method='simpsons', fast=False):
        
        if self.kappa is None:
            self.alpha_x = None
            self.alpha_y = None  
        elif len(self.kappa.shape) == 2:
            
            #create empty arrays to be filled with x,y components of alpha
            alpha_x = np.empty([self.NX_image,self.NY_image], float)
            alpha_y = np.empty([self.NX_image,self.NY_image], float)
            start = time()
            if method == 'simpsons':            
            #double for loop to get each point in array
                
                for i in range(len(alpha_x[:,0])):
                    for j in range(len(alpha_x[0,:])):
#                    '''calculate deflection angles using simpsons rule.  Uses 
#                    xgrid, ygrid to determine dx and dy.  Very accurate, but
#                    can take > 0.1 s per integral for large grids.
#                    '''                    
                        alpha_x[i,j] = simps(simps(1/np.pi *self.kappa  \
                        * (self.image_x[i,j]-self.x)/((self.image_x[i,j]\
                        -self.x)**2+(self.image_y[i,j]-self.y)**2),x =  \
                        self.xgrid),x=self.ygrid)
                        
                        alpha_y[i,j] = simps(simps(1/np.pi *self.kappa * \
                        (self.image_y[i,j]-self.y)/((self.image_x[i,j]-\
                        self.x)**2+(self.image_y[i,j]-self.y)**2),x = \
                        self.xgrid),x=self.ygrid)
                     
            elif method == 'rectangles':
#                '''Compute integrals by approximating pixels as point masses
#                with position equivalent to their x,y coordinates.  In 
#                principle this is less accurate than simpson's rule, but
#                it is significantly faster.
#                '''
                if fast is True:
                    
                    K = self.kappa/np.pi*self.pixscale**2
                    for i in range(len(alpha_y[:,0])):
                        K2 = K*(self.image_y[i,0]-self.y)
                        y2 = (self.image_y-self.y)**2
                        for j in range(len(alpha_y[0,:])):
                            alpha_y[i,j] = np.sum(K2/(y2 \
                            + (self.image_x[i,j]-self.x)**2))
                    for j in range(len(alpha_x[0,:])):
                        K2 = K*(self.image_x[0,j]-self.x)
                        x2 = (self.image_x[0,j]-self.x)**2
                        for i in range(len(alpha_x[:,0])):
                            alpha_x[i,j] = np.sum(K2/(x2 \
                            +(self.image_y[i,j]-self.y)**2))
                else:
                    for i in range(len(alpha_x[:,0])):
                        for j in range(len(alpha_x[0,:])):
                    
                            alpha_x[i,j] =1/np.pi * np.sum(self.kappa * (self.image_x[i,j]-self.x)/((self.image_x[i,j]-self.x)**2+(self.image_y[i,j]-self.y)**2)*self.pixscale**2)
                            alpha_y[i,j] =1/ np.pi * np.sum(self.kappa * (self.image_y[i,j]-self.y)/((self.image_x[i,j]-self.x)**2+(self.image_y[i,j]-self.y)**2)*self.pixscale**2)
            
            elif method == 'trapezoidal':  
                # Compromise between simpsons rule and rectangle rule
                # Creates weights array to avoid overhead in the for loops.
                
                weights = np.zeros([self.kappa.shape[0],self.kappa.shape[1]],float)
                weights[0,0] = 1.0
                weights[-1,0] = 1.0
                weights[0,-1] = 1.0
                weights[-1,-1] = 1.0
                weights[0:,1:-1] =2.0
                weights[1:-1,0:] +=2.0
                K = 1.0/(np.pi*4.0)*weights*self.kappa*self.pixscale**2
                
                if fast is True:
                    for i in range(len(alpha_y[:,0])):
                        K2 = K*(self.image_y[i,0]-self.y)
                        y2 = (self.image_y[i,0]-self.y)**2
                        
                        for j in range(len(alpha_y[0,:])):
                            alpha_y[i,j] = np.sum(K2/(y2 \
                            +(self.image_x[i,j]-self.x)**2))
                    for j in range(len(alpha_x[0,:])):
                        K2 = K*(self.image_x[0,j]-self.x)
                        x2 = (self.image_x[0,j]-self.x)**2
                        
                        for i in range(len(alpha_x[:,0])):
                            alpha_x[i,j] = np.sum(K2/(x2+ \
                            (self.image_y[i,j]-self.y)**2))
                        
                
                
                else:
                    for i in range(len(alpha_x[:,0])):
                        for j in range(len(alpha_x[0,:])):
                            alpha_x[i,j] = np.sum(K*(self.image_x[i,j]-\
                            self.x)/((self.image_x[i,j]-self.x)**2 + \
                            (self.image_y[i,j]-self.y)**2))
                        
                            alpha_y[i,j] = np.sum(K*(self.image_y[i,j] \
                            -self.y)/((self.image_x[i,j]-self.x)**2 + \
                            (self.image_y[i,j]-self.y)**2))
                
                    
            elif method == 'FFT':
                '''
                Calculate the lensing potential using an FFT, and take the derivative to get the deflection angles.
                '''
                # pad the arrays to assume vacuum boundary conditions.
                k_pad = np.zeros([self.kappa.shape[0]*3,self.kappa.shape[1]*3])
                k_pad[self.kappa.shape[0]:2*self.kappa.shape[0],self.kappa.shape[1]:2*self.kappa.shape[1]] =self.kappa
                
                k_pad_ft = np.fft.fft2(k_pad)
                fx = np.fft.fftfreq(3*self.x.shape[0],self.pixscale)
                fy = np.fft.fftfreq(3*self.x.shape[0],self.pixscale)
                fx,fy = np.meshgrid(fx,fy)
                psif = k_pad_ft /(2*np.pi**2*(fx**2+fy**2))
                psif[0,0] = 0
                
                psi = np.fft.ifft2(psif)[self.kappa.shape[0]:2*self.kappa.shape[0],self.kappa.shape[1]:2*self.kappa.shape[1]]
                alpha_y, alpha_x = np.gradient(psi.real,self.pixscale)
                alpha_y *= -1
                alpha_x *= -1
                
                # keep the oversampling and pixel size ratio
                xmin = self.NX // 2 - self.n2 * ( self.NX // ( 2 * self.n ) )
                xmax = self.NX // 2 + self.n2 * ( self.NX // ( 2 * self.n ) )
                ymin = self.NY // 2 - self.n2 * ( self.NY // ( 2 * self.n ) )
                ymax = self.NY // 2 + self.n2 * ( self.NY // ( 2 * self.n ) )
                
                alpha_x = alpha_x[xmin:xmax:self.n2,ymin:ymax:self.n2]
                alpha_y = alpha_y[xmin:xmax:self.n2,ymin:ymax:self.n2]
                
                print 'resetting image (x,y) coordinates'
                self.image_x = self.x[xmin:xmax:self.n2,ymin:ymax:self.n2]
                self.image_y = self.y[xmin:xmax:self.n2,ymin:ymax:self.n2]
            
                        
            else:
                print('you must choose a valid method of deflection')
                print(' your deflection angles will not be correct ')
            
            self.alpha_x = alpha_x
            self.alpha_y = alpha_y
            stop = time()
            print('Elapsed seconds during calculation:',stop-start)
        else:
            raise Exception("Can't do integral.  your kappa map must be 2-D .\n")  
        
        return
    
# ----------------------------------------------------------------------    
    
    def plot(self,mapname, caustics=True,critical_curves=True,Ngridlines=100,xlim=None,ylim=None,figscale=1):    
        '''
        Plot the given map as a nice colorscale image, with contours if need be.
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
            options['extent'] = (np.min(self.x),np.max(self.x),\
                                 np.min(self.y),np.max(self.y))
                           
        elif mapname == "alpha":
            img1 = self.alpha_x
            img2 = self.alpha_y
            levels = np.arange(-0.5,0.5,0.1)
            options = dict(interpolation='nearest',\
                           origin='lower',\
                           vmin=-1.0, \
                           vmax=1.0)
            options['extent'] = (np.min(self.image_x),np.max(self.image_x),\
                                 np.min(self.image_y),np.max(self.image_y))
        elif mapname == "alpha_x":
            img = self.alpha_x
            levels = np.arange(-0.5,0.5,0.1)
            options = dict(interpolation='nearest',\
                           origin='lower',\
                           vmin=-1.0, \
                           vmax=1.0)                           
            options['extent'] = (np.min(self.image_x),np.max(self.image_x),\
                                 np.min(self.image_y),np.max(self.image_y))
            
        elif mapname == "alpha_y":
            img = self.alpha_y
            levels = np.arange(-0.5,0.5,0.1)
            options = dict(interpolation='nearest',\
                           origin='lower',\
                           vmin=-1.0, \
                           vmax=1.0)
            options['extent'] = (np.min(self.image_x),np.max(self.image_x),\
                                 np.min(self.image_y),np.max(self.image_y))
                           
        elif mapname == "lensed image":
            img = self.image
            #This statement sets contrast based on the type of image.
            if (np.max(self.image)-np.average(self.image))/np.average(self.image) < 1:
                options = dict(origin='lower',\
                               vmin=np.min(self.image)*0.95, \
                               vmax=np.max(self.image)*0.95)
            else:
                options = dict(origin='lower',\
                               vmin=-0.2, \
                               vmax=np.max(self.image)*0.95)
            options['extent'] = (np.min(self.image_x),np.max(self.image_x),\
                                 np.min(self.image_y),np.max(self.image_y))                               
                        
        elif mapname == "non-lensed image":
            img = self.source.intensity
            #same as for lensed image, here we guess for the contrast.
            if (np.max(self.image)-np.average(self.image))/np.average(self.image) < 1:
                options = dict(interpolation='nearest',\
                               origin='lower',\
                               vmin=np.min(self.source.intensity)*0.95, \
                               vmax=np.max(self.source.intensity)*0.95)
            else:
                options = dict(interpolation='nearest',\
                               origin='lower',\
                               vmin=-0.2, \
                               vmax=np.max(self.source.intensity)*0.95)
            options['extent'] = (np.min(self.source.beta_x),np.max(self.source.beta_x),\
                                 np.min(self.source.beta_y),np.max(self.source.beta_y))
        else:
             raise ValueError("unrecognized map name %s" % mapname)
        
        # set figure up for multiple plots:
        if mapname == "alpha":
            px,py = 2*figscale,1*figscale       
        else:
            px,py = 1*figscale,1*figscale
        
        
        figprops = dict(figsize=(5*px,5*py), dpi=128)
        adjustprops = dict(left=0.1,\
                           bottom=0.1,\
                           right=0.95,\
                           top=0.95,\
                           wspace=0.1,\
                           hspace=0.1)
        
        
        # 2) The viridis map is linear grayscale on a BW printer
        options['cmap'] = plt.get_cmap('viridis')

        
        # Start the figure:
        fig = plt.figure(**figprops)
        fig.subplots_adjust(**adjustprops)
        plt.clf()

        # Plot a colored pixel map.  Options determined for each type of map
        if mapname == "kappa":
            plt.imshow(img, **options)
            cbar = plt.colorbar(shrink = 0.75)
            cbar.set_label('$ \kappa $ / dimensionless',fontsize=12)
            plt.contour(self.x, self.y, img, levels,colors=('k',))
            plt.xlabel('x / arcsec')
            plt.ylabel('y / arcsec')
        elif mapname =="alpha":
            plt.subplot(121)
            plt.imshow(img1,**options)
            cbar = plt.colorbar(shrink = 0.75)
            cbar.set_label(r'$ \alpha_{x} $ / arcsec',fontsize=12)
            plt.xlabel('x / arcsec')
            plt.ylabel('y / arcsec')
            plt.subplot(121).set_aspect('equal')
            plt.subplot(122)
            plt.xlabel('x / arcsec')
            plt.subplot(122).set_aspect('equal')
            plt.imshow(img2, **options)
            cbar2 = plt.colorbar(shrink = 0.75)    
            cbar2.set_label(r'$ \alpha_{y} $ / arcsec',fontsize=12)
            fig.tight_layout()
        elif mapname == "alpha_x":
            plt.imshow(img, **options)
            cbar = plt.colorbar(shrink = 0.75)
            cbar.set_label(r'$\alpha_{x} $ / arcsec',fontsize=12)

        elif mapname == "alpha_y":
            plt.imshow(img, **options)
            cbar = plt.colorbar(shrink = 0.75)
            cbar.set_label(r'$\alpha_{y} $ / arcsec',fontsize=12)

        elif mapname == "lensed image":
            #plot same as everything else if achromatic
            if len(img.shape)==2:
                plt.imshow(img,**options)
                cbar = plt.colorbar(shrink = 0.75)
                cbar.set_label('Flux / arbitrary units',fontsize=12)
            elif (img.shape[0])==3:
                options['vmin']=None
                options['vmax']=None
                img_new = np.empty([img.shape[1],img.shape[2],img.shape[0]],float)
                for i in range(img.shape[0]):
                    img_new[:,:,i] = img[i,:,:]
                plt.imshow(img_new, **options)
                
            else:
                raise Exception("Cannot plot multiwavelength images yet.\n")
                
            if critical_curves is True:
                jxy,jxx = np.gradient(self.beta_x)
                jyy,jyx = np.gradient(self.beta_y)
                A = jxx*jyy-jxy*jyx
                
                plt.contour(self.image_x,self.image_y,A,levels=[0],colors='r')
            plt.xlabel('x / arcsec')
            plt.ylabel('y / arcsec')
        
        elif mapname == "non-lensed image":
            
            if len(img.shape)==2:
                plt.imshow(img,**options)
                cbar = plt.colorbar(shrink = 0.75)
                cbar.set_label('Flux / arbitrary units', fontsize=12)
            elif (img.shape[0])==3:
                options['vmin']=None
                options['vmax']=None
                img_new = np.empty([img.shape[1],img.shape[2],img.shape[0]],float)
                for i in range(img.shape[0]):
                    img_new[:,:,i] = img[i,:,:]
                plt.imshow(img_new, **options)
            else:
                raise Exception("Cannot plot many (>3) wavelength images yet.\n")
            plt.xlabel('x / arcsec')
            plt.ylabel('y / arcsec')
            
            if caustics is True:
                
                jxy,jxx = np.gradient(self.beta_x)
                jyy,jyx = np.gradient(self.beta_y)
                A = jxx*jyy-jyx*jxy
                plt.contour(self.beta_x,self.beta_y,A,levels=[0],colors='r')
                
                #pixstep1 = self.beta_x.shape[0]//Ngridlines
                #pixstep2 = self.beta_x.shape[1]//Ngridlines
                #for i in range(Ngridlines):
                #    plt.plot(self.beta_x[pixstep1*i,:],self.beta_y[pixstep1*i,:],'r-',linewidth=0.1)
                #    plt.plot(self.beta_x[:,pixstep2*i],self.beta_y[:,pixstep2*i],'r-',linewidth=0.1)
                
                #plt.scatter(self.beta_x[::10,::10],self.beta_y[::10,::10], s=0.001)
                plt.xlim(np.min(self.source.beta_x),np.max(self.source.beta_x))
                plt.ylim(np.min(self.source.beta_y),np.max(self.source.beta_y))
            else:
                pass
            if xlim is not None:
                plt.xlim(xlim[0],xlim[1])
                plt.ylim(ylim[0],ylim[1])
        else:
            pass
                

        # If we're in a notebook, display the plot. 
        # Otherwise, make a PNG.
        try:
            __IPYTHON__
            plt.show()
        except NameError:
            pngfile = mapname+'.png'
            plt.savefig(pngfile)
            print "Saved plot to "+pngfile
            
        return fig


# ----------------------------------------------------------------------

    def write_kappa_to( self, fitsfile="kappa_map.fits"):
        
        hdu = fits.PrimaryHDU(self.kappa)
        
        hdu.header['CDELT1'] = self.pixscale / 3600.0
        hdu.header['CDELT2'] = self.pixscale / 3600.0
        hdu.header['NPADDING'] = self.n
        hdu.header['NPAD2']= self.n2
        
        
        hdu.writeto(fitsfile,clobber=True)
       
        return

# ----------------------------------------------------------------------

    def write_image_to(self, fitsfile="lensed_image.fits"):
        '''
        Write a lensed image to a fits file to feed into an 
        observing simulator.
        '''        
        
        hdu = fits.PrimaryHDU(self.image)
        hdu.header['CDELT1'] = self.pixscale / 3600.0
        hdu.header['CDELT2'] = self.pixscale / 3600.0
        hdu.writeto(fitsfile)
        
        return
        
# ----------------------------------------------------------------------

    def raytrace(self):
        '''
        Create observed image grid, then use lens equation to find
        angles in the source plane.  Use bilinear interpolation to
        get intensity at each observed image pixel.
        
        This function works for 2 dimensional (single color) and
        3 dimensional (multicolor) images.  Interpolation is done
        for each color channel image separately and independently.
        '''       
        if self.source is None: 
            raise Exception("Can't do raytracing yet.\n")  
            
        else:
            
            #  give each pixel in the image an x,y position, should be same as image_x, image_y
            self.theta_x = np.copy(self.image_x)
            self.theta_y = np.copy(self.image_y)            
            
            #theta_x = np.arange(-(self.NX//self.n)/2.0,(self.NX//self.n)/2.0,1.0)*self.pixscale+self.pixscale+self.pixel_offset
            #theta_y = np.arange(-(self.NY//self.n)/2.0,(self.NY//self.n)/2.0,1.0)*self.pixscale+self.pixscale+self.pixel_offset
            #self.theta_x,self.theta_y = np.meshgrid(theta_x,theta_y)
            
            #Find the corresponding angles in the source plane              
            self.beta_x = self.theta_x-self.alpha_x
            self.beta_y = self.theta_y-self.alpha_y            
            
            # single wavelength
            if len(self.source.intensity.shape) ==2:
                
                #  first create empty image with dimensions NX, NY
                #  (we should make this more general later)  
                self.image = np.empty([self.NY//self.n,self.NX//self.n],float)
            
                #create bilinear interpolation function (assumes uniform grid of x,y)
                f_interpolation = interpolate.RectBivariateSpline(self.source.beta_y[:,0],self.source.beta_x[0,:],self.source.intensity,kx=1,ky=1)            
            
                #interpolate for observed intensity at each angle            
                for i in range(len(self.image[:,0])):
                    for j in range(len(self.image[0,:])):                    
                       self.image[i,j] = f_interpolation(self.beta_y[i,j],self.beta_x[i,j])
                
            else:   #multiwavelength data cube
                self.image = np.empty([self.source.Naxes,self.NY//self.n,self.NX//self.n], float)
                
                for i in range(self.source.Naxes):
                    f_interpolation = interpolate.RectBivariateSpline(self.source.beta_y[:,0],self.source.beta_x[0,:],self.source.intensity[i,:,:],kx=1,ky=1)
                    
                    self.image[i,:,:] = f_interpolation.ev(self.beta_y.flatten(),self.beta_x.flatten()).reshape([self.NY//self.n,self.NX//self.n])
                    
#                    for k in range(self.NX//self.n):
#                        for j in range(self.NY//self.n):
#                            self.image[i,j,k] = f_interpolation(self.beta_y[j,k],self.beta_x[j,k])
                            
        return

# ---------------------------------------------------------------------

    def __add__(self,right):
        '''
        Add two gravitational lenses together to make a third lens.
        In order to do this, both lenses must have kappa maps with
        the same angular dimensions and pixel sizes.
        '''        
        
        #raise Exception("Cannot add lenses yet.\n")
        if issubclass(type(self),GravitationalLens) is False and issubclass(type(right),GravitationalLens) is False :
            raise TypeError('unsupported operand type(s)')
        assert len(self.kappa.shape) == len(right.kappa.shape)
        assert self.kappa.shape == right.kappa.shape
        assert abs(self.pixscale - right.pixscale) <10**-10
        assert self.n == right.n
        assert self.n2 ==right.n2
        
        newLens = GravitationalLens(self.Zd,self.Zs)
        newLens.NX,newLens.NY = self.kappa.shape
        newLens.pixscale = self.pixscale
        newLens.n = self.n
        newLens.n2 = self.n2        
        
        # Set up a new pixel grid to go with this new kappa map:
        newLens.setup_grid()        
        
        if self.kappa is not None and right.kappa is not None:
            newLens.kappa = self.kappa+right.kappa
        if self.alpha_x is not None and right.alpha_x is not None:
            newLens.alpha_x = self.alpha_x+right.alpha_x
            newLens.alpha_y = self.alpha_y+right.alpha_y
        
        
        return(newLens)
        
# ----------------------------------------------------------------------
        
    def __sub__(self,right):
        '''
        Subtract one lens from another to make a third lens.  This
        is mostly useful for computing errors on deflection angles,
        but functionalities could be increased to include 
        differences in kappa or lensed images.
        '''
        #raise Exception("Cannot subtract lenses yet.\n")
        if issubclass(type(self),GravitationalLens) is False and issubclass(type(right),GravitationalLens) is False:
            raise TypeError('unsupported operand type(s)')
        assert len(self.kappa.shape) == len(right.kappa.shape)
        assert self.kappa.shape == right.kappa.shape
        assert abs(self.pixscale - right.pixscale) <10**-10
        assert self.n == right.n
        assert self.n2 == right.n2
        
        newLens = GravitationalLens(self.Zd,self.Zs)
        newLens.NX,newLens.NY = self.kappa.shape
        newLens.pixscale = self.pixscale
        newLens.n = self.n    
        newLens.n2 = self.n2
        
        # Set up a new pixel grid to go with this new kappa map:
        newLens.setup_grid()        
        
        if self.kappa is not None and right.kappa is not None:
            newLens.kappa = self.kappa -right.kappa
        if self.alpha_x is not None and right.alpha_x is not None:
            newLens.alpha_x = self.alpha_x - right.alpha_x
            newLens.alpha_y = self.alpha_y - right.alpha_y
        
        return(newLens)
        
# ======================================================================

if __name__ == '__main__':

    lens = GravitationalLens(0.4,1.5)
    
    print "Difference in angular diameter distances: ",lens.Ds - lens.Dd
    print "  cf. Dds = ", lens.Dds
    print "Critical density = ",lens.SigmaCrit

    lens.read_kappa_from("/Users/wmorning/Research/EvilLens/examples/test_kappa.fits")
    lens.plot("kappa")
            
# ======================================================================
