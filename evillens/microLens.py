from astropy import units, constants
import numpy as np
import evillens as evil
from scipy import interpolate, spatial
import matplotlib.pyplot as plt

# ========================================================================

class MicroLens(evil.GravitationalLens):
    '''
    A microlens in the galaxy (MACHO or star with exoplanet)
    '''
    def __init__(self, *args, **kwargs):
        super(MicroLens, self).__init__(*args, **kwargs)
        self.M = 1.0*units.solMass
        self.centroid = [0.00,0.00]
        self.src_cent = [0.0,0.0]
        self.Dd = 5*units.kpc 
        self.Ds = 10*units.kpc
        self.Dds= self.Ds-self.Dd
        self.thetaE = self.compute_thetaE(self.M,self.Dd,self.Ds,self.Dds)
        self.setup_source([0.0,0.0],100,100,10**-6)
        self.simulation_setup = False
        return
        
    def compute_thetaE(self,Mass,Dd,Ds,Dds):
        thetaE = ((4*np.pi*constants.G*self.M/constants.c**2 * Dds /(Dd*Ds))**(1/2.0)).decompose().value*3600*180/np.pi
        return thetaE
        
    def setup_source(self,src_cent,NX,NY,src_L):
        '''
        source grid has center and size
        '''
        self.src_NX = NX
        self.src_NY = NY
        self.src_cent = src_cent
        beta_x = np.linspace(src_cent[0]-src_L/2.0,src_cent[0]+src_L/2.0,NX)
        beta_y = np.linspace(src_cent[1]-src_L/2.0,src_cent[1]+src_L/2.0,NY)
        self.src_beta_x,self.src_beta_y = np.meshgrid(beta_x,beta_y)
        return
    def build_source(self,I0,size,cent):
        '''
        build source.  assumes gaussian light profile.  Width is in solar radii
        '''
        self.src_size = (size*constants.R_sun/self.Ds).decompose().value * 3600*180/np.pi
        sigma = self.src_size
        self.setup_source(cent,self.src_NX,self.src_NY,10*self.src_size)
        self.src_intensity = I0/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*((self.src_beta_x-cent[0])**2+(self.src_beta_y-cent[1])**2)/sigma**2)
        return
        
        
    def setup_grids(self,NX,NY):
        '''
        Microlens has two image grids, at predicted positions of source.
        '''
        self.NX = NX
        self.NY = NY

        src_theta = np.arctan2(self.src_cent[1]-self.centroid[1],self.src_cent[0]-self.centroid[0])
        src_r     = np.sqrt((self.centroid[0]-self.src_cent[0])**2+(self.src_cent[1]-self.centroid[1])**2)
        r1 = (src_r+np.sqrt(src_r**2+4*self.thetaE**2))/2.0
        r2 = (src_r-np.sqrt(src_r**2+4*self.thetaE**2))/2.0
        
        cent1 = [r1*np.cos(src_theta)+self.centroid[0],r1*np.sin(src_theta)+self.centroid[1]]
        cent2 = [r2*np.cos(src_theta)+self.centroid[0],r2*np.sin(src_theta)+self.centroid[1]]
        
        im_x1 = np.linspace(cent1[0]-50*self.src_size,cent1[0]+50*self.src_size,NX)
        im_y1 = np.linspace(cent1[1]-50*self.src_size,cent1[1]+50*self.src_size,NY)
        im_x2 = np.linspace(cent2[0]-50*self.src_size,cent2[0]+50*self.src_size,NX)
        im_y2 = np.linspace(cent2[1]-50*self.src_size,cent2[1]+50*self.src_size,NY)
        
        self.imx1,self.imy1 = np.meshgrid(im_x1,im_y1)
        self.imx2,self.imy2 = np.meshgrid(im_x2,im_y2)
        
        return cent1,cent2
    
    def deflect(self):
        '''
        compute deflection angles for both image grids.
        '''
        self.alpha_x1 = self.thetaE**2*(self.imx1-self.centroid[0])/((self.imx1-self.centroid[0])**2+(self.imy1-self.centroid[1])**2)
        self.alpha_y1 = self.thetaE**2*(self.imy1-self.centroid[1])/((self.imx1-self.centroid[0])**2+(self.imy1-self.centroid[1])**2)
        self.alpha_x2 = self.thetaE**2*(self.imx2-self.centroid[0])/((self.imx2-self.centroid[0])**2+(self.imy2-self.centroid[1])**2)
        self.alpha_y2 = self.thetaE**2*(self.imy2-self.centroid[1])/((self.imx2-self.centroid[0])**2+(self.imy2-self.centroid[1])**2)
        
        return
        
    def add_exoplanet(self,Mp,cent):
        '''
        add exoplanet to deflection angles
        '''
        tEp = ((4*np.pi*constants.G*Mp*units.solMass/constants.c**2 * self.Dds /(self.Dd*self.Ds) )**(1/2.0)).decompose().value*3600*180/np.pi
        self.alpha_x1 += tEp**2*(self.imx1-cent[0])/((self.imx1-cent[0])**2+(self.imy1-cent[1])**2)
        self.alpha_y1 += tEp**2*(self.imy1-cent[1])/((self.imx1-cent[0])**2+(self.imy1-cent[1])**2)
        self.alpha_x2 += tEp**2*(self.imx2-cent[0])/((self.imx2-cent[0])**2+(self.imy2-cent[1])**2)
        self.alpha_y2 += tEp**2*(self.imy2-cent[1])/((self.imx2-cent[0])**2+(self.imy2-cent[1])**2)
        
        #self.alpha_x1[(np.sqrt((self.imx1-cent[0])**2+(self.imy1-cent[1])**2)<0.05*tEp)] *=0
        #self.alpha_y1[(np.sqrt((self.imx1-cent[0])**2+(self.imy1-cent[1])**2)<0.05*tEp)] *=0
            
        
    def raytrace(self):
        '''
        map source image back to lens plane to create two lensed images
        '''
        self.beta_x1 = self.imx1 - self.alpha_x1
        self.beta_y1 = self.imy1 - self.alpha_y1
        self.beta_x2 = self.imx2 - self.alpha_x2
        self.beta_y2 = self.imy2 - self.alpha_y2
        
        self.im1 = np.zeros(self.imx1.shape) 
        self.im2 = np.zeros(self.imy1.shape)
        
        f_interpolation = interpolate.RectBivariateSpline(self.src_beta_y[:,0],self.src_beta_x[0,:],self.src_intensity,kx=1,ky=1) 
        
        for i in range(len(self.im1[:,0])):
            for j in range(len(self.im1[0,:])):                    
               self.im1[i,j] = f_interpolation(self.beta_y1[i,j],self.beta_x1[i,j])
               self.im2[i,j] = f_interpolation(self.beta_y2[i,j],self.beta_x2[i,j])
        
        #self.im1 = f_interpolation(self.beta_y1.flatten(),self.beta_x1.flatten()).reshape(self.NY,self.NX)
        #self.im2 = f_interpolation(self.beta_y2.flatten(),self.beta_x2.flatten()).reshape(self.NY,self.NX)           
        
        
    def get_magnification(self):
        self.img_pix_x = (np.max(self.imx1)-np.min(self.imx1))/float(self.NX)
        self.img_pix_y = (np.max(self.imy1)-np.min(self.imy1))/float(self.NY)
        
        self.src_pix_x = (np.max(self.src_beta_x)- np.min(self.src_beta_x)) / float(self.src_NX)
        self.src_pix_y = (np.max(self.src_beta_y)- np.min(self.src_beta_y)) / float(self.src_NY)
        
        img_Lum = np.sum(self.im1)*self.img_pix_x*self.img_pix_y+np.sum(self.im2)*self.img_pix_x*self.img_pix_y
        src_Lum = np.sum(self.src_intensity)*self.src_pix_x*self.src_pix_y
        
        return img_Lum/src_Lum
        
    def setup_simulation(self,M,Dd,b,Ds,srcL,q=None,d=None,phi=None,Nsamples=100,src_pix=100,lens_pix=100):
        '''
        Setup the full microlensing simulation.  Accepts the following parameters:
        - Lens Mass M
        - Lens Distance Dd
        - Impact paramter b in units of Einstein Radius
        - Source Distance Ds
        - Exoplanet mass ratio q (default is none).  List if multiple exoplanets.
        - Exoplanet Separation from star (in Einstein Radii)
        - Angle between source trajectory and lens-planet vector.
        - Size of source, in solar radii
        
        - Number of sample points in the light curve
        
        Note that the source is always assumed to be travelling in the x-direction.
        '''
        
        # first setup lens parameters
        self.M = M*units.solMass
        self.centroid = [0.00,0.00]
        self.Dd = Dd*units.kpc 
        self.Ds = Ds*units.kpc
        self.Dds= self.Ds-self.Dd
        self.thetaE = self.compute_thetaE(self.M,self.Dd,self.Ds,self.Dds)
        self.b = b*self.thetaE
        
        # second, compute source trajectory
        self.y = b*np.ones(Nsamples)*self.thetaE
        self.x = np.linspace(-1.5,1.5,Nsamples)*self.thetaE
        
        if q is not None:
            self.q = q
        if d is not None:
            self.d = np.multiply(d ,self.thetaE)
        if phi is not None:
            self.phi = phi
        self.srcL = srcL
        self.src_pix=src_pix
        self.lens_pix=lens_pix
        self.setup_source([0.0,0.0],self.src_pix,self.src_pix,self.srcL)
        if (q is not None) & (d is not None) & (phi is not None):
            self.Exoplanets = True
            self.pos = [[self.d[i]*np.cos(phi[i]),self.d[i]*np.sin(phi[i])] for i in range(len(d))]
            self.Mp = [(q[i]*self.M).value for i in range(len(q))]
        else:
            self.Exoplanets = False
            
        
        self.simulation_setup = True
        
    def run_simulation(self,x=None,y=None, animate=False, folder=None):
        
        if self.simulation_setup != True:
            print "need to setup simulation"
            return
        else:
            if x is not None:
                self.x = x
                self.y = y
            
            # create list to store magnification 
            magnification = []
            Img1_x = []
            Img1_y = []
            Img2_x = []
            Img2_y = []
            
            # start simulation
            for i in range(len(self.x)):
                self.build_source(1,self.srcL,[self.x[i],self.y[i]])
                cent1, cent2 = self.setup_grids(self.lens_pix,self.lens_pix)
                self.deflect()
                if self.Exoplanets is True:
                    for j in range(len(self.Mp)):
                        self.add_exoplanet(self.Mp[j],self.pos[j])
                self.raytrace()
    
                magnification.append(self.get_magnification())
                Img1_x.append(cent1[0])
                Img1_y.append(cent1[1])
                Img2_x.append(cent2[0])
                Img2_y.append(cent2[1])
                
                if animate == True:
                    plt.clf()
                    fig = plt.figure()
                    plt.subplot2grid([2,3],[0,0],1,1)
                    plt.imshow(self.im1,animated=True,vmin=0,vmax=np.max(self.src_intensity),origin='lower',cmap='hot',extent=(np.min(self.imx1),np.max(self.imx1),np.min(self.imy1),np.max(self.imy1)))
                    plt.xticks([])
                    plt.yticks([])
                    if self.Exoplanets is True:
                        for j in range(len(self.Mp)):
                            plt.plot(self.pos[j][0],self.pos[j][1],'wo')
                    plt.xlim(np.min(self.imx1),np.max(self.imx1))
                    plt.ylim(np.min(self.imy1),np.max(self.imy1))
                            
                    
                    plt.subplot2grid([2,3],[0,1],1,1)
                    plt.plot([0],[0],'ko',label = 'star')
                    einrad=plt.Circle((0,0),self.thetaE,color='k',fill=False)
                    if self.Exoplanets is True:
                        for j in range(len(self.Mp)):
                            plt.plot(self.pos[j][0],self.pos[j][1],'bo',label='planets')
                    plt.plot(Img1_x,Img1_y,'k-')
                    plt.plot(Img2_x,Img2_y,'k-')
                    plt.plot(self.x[:i],self.y[:i],'k--')
                    #plt.plot(Img2_x[-1],Img2_y[-1],'co',label='Image Position')
                    #plt.plot(Img1_x[-1],Img1_y[-1],'co',label='Image Position')
                    plt.plot(self.x[i],self.y[i],'mo')
                    plt.imshow(self.im1,animated=True,vmin=0,vmax=np.max(self.src_intensity),origin='lower',cmap='hot',extent=(np.min(self.imx1),np.max(self.imx1),np.min(self.imy1),np.max(self.imy1)))
                    plt.imshow(self.im2,animated=True,vmin=0,vmax=np.max(self.src_intensity),origin='lower',cmap='hot',extent=(np.min(self.imx2),np.max(self.imx2),np.min(self.imy2),np.max(self.imy2)))
       #             img1 = plt.Rectangle([Img1_x[-1]-50*self.src_size,Img1_y[-1]-50*self.src_size],100*self.src_size,100*self.src_size,fill=False,color='k')
        #            img2 = plt.Rectangle([Img2_x[-1]-50*self.src_size,Img2_y[-1]-50*self.src_size],100*self.src_size,100*self.src_size,fill=False,color='k')
                    fig.gca().add_artist(einrad)
         #           fig.gca().add_artist(img1)
          #          fig.gca().add_artist(img2)
                    plt.xlim(-1.5*self.thetaE,1.5*self.thetaE)
                    plt.ylim(-1.5*self.thetaE,1.5*self.thetaE)
                    plt.yticks([])
                    plt.xticks([])
                    plt.subplot2grid([2,3],[0,2],1,1)
                    plt.imshow(self.im2,animated=True,vmin=0,vmax=np.max(self.src_intensity),origin='lower',cmap='hot',extent=(np.min(self.imx2),np.max(self.imx2),np.min(self.imy2),np.max(self.imy2)))
                    plt.xticks([])
                    plt.yticks([])
                    if self.Exoplanets is True:
                        for j in range(len(self.Mp)):
                            plt.plot(self.pos[j][0],self.pos[j][1],'wo',label='planets')
                    plt.xlim(np.min(self.imx2),np.max(self.imx2))
                    plt.ylim(np.min(self.imy2),np.max(self.imy2))
                    
                    plt.subplot2grid([2,3],[1,0],3,3)
                    plt.plot(self.x[:len(magnification)]/self.thetaE,magnification,'k-')
                    plt.plot(self.x[len(magnification)-1]/self.thetaE,magnification[-1],'ko')
                    plt.xlim(-1.25,1.25)
                    plt.ylim(1,np.max([6,np.max(magnification)]))
                    plt.ylabel('Magnification',fontsize=18)
                    plt.xlabel(r'Time ($t_{E}$)',fontsize=18)
                    plt.savefig(folder+'frame_{0}.png'.format(i))
                    
        
        return magnification ,Img1_x,Img1_y,Img2_x,Img2_y
                
            
    