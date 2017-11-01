import numpy as np
import evillens as evil

# ======================================================================

class SersicSource(evil.Source):
    '''
    A sersic source profile.  Adds a simple build_source function
    '''
    def __init__(self, *args, **kwargs):
        super(SersicSource, self).__init__(*args, **kwargs)
        
        self.Flux = 0
        self.position = [0,0]
        self.q = 1.0
        self.angle = 0.0
        self.n = 4
        self.reff = 1
        
        return    
        
# ----------------------------------------------------------------------

    def Build_Source(self,Flux=0,position=[0,0],q=1.0,angle=0.0,n=4.,reff=1):
    
        """
        Build source intensity using sersic profile
    
        Parameters:
    
        Flux:       Total flux of the source (in mJy)
    
        position:   [x,y] position of the source (in arcseconds)
    
        q:          Axis ratio of the source
    
        angle:      Angle of major axis counterclockwise from the positive x-axis
                    (in degrees)
    
        n:          Sersic Index.
    
        reff:       The e-folding scale length of the source.    
            
        """
        if Flux is not None:
            self.Flux = Flux
        if position is not None:
            self.position = position
        if q is not None:
            self.q = q
        if angle is not None:
            self.angle = angle * np.pi / 180.
        if n is not None:
            self.n = n
        if reff is not None:
            self.reff = reff
        
        # convert to rotated coordinates
        xp = np.cos(self.angle)*(self.beta_x-self.position[0]) + np.sin(self.angle)*(self.beta_y-self.position[1])
        yp =-np.sin(self.angle)*(self.beta_x-self.position[0]) + np.cos(self.angle)*(self.beta_y-self.position[1])
    
        # convert to elliptical radius
        r = np.sqrt(self.q*xp**2 + yp**2/self.q)
    
        # Sersic is 1-d function of radius
        self.intensity = np.exp(-(r / reff)**(1/self.n))
    
        # Normalize to get flux
        self.intensity *= self.Flux/np.sum(self.intensity)
    
        return
        
# ----------------------------------------------------------------------
    
    def build_from_clumps(self):
        return

# ----------------------------------------------------------------------

    def build_sersic_clumps(self):
        return
    
# ======================================================================

class GaussianSource(evil.Source):
    """
    A gaussian source profile
    """
    def __init__(self,*args,**kwargs):
        super(GaussianSource,self).__init__(*args,**kwargs)
        
        self.Flux = 0
        self.position = [0,0]
        self.q = 1.0
        self.angle = 0.0
        self.sigma = 1.0
        return
        
# ----------------------------------------------------------------------

    def Build_Source(self,Flux=0,position=[0,0],q=1.0,angle=0.0,sigma=1.0):
        if Flux is not None:
            self.Flux = Flux
        if position is not None:
            self.position = position
        if q is not None:
            self.q = q
        if angle is not None:
            self.angle = angle * np.pi / 180.
        if sigma is not None:
            self.sigma = sigma
    
        # convert to rotated coordinates
        xp = np.cos(self.angle)*(self.beta_x-self.position[0]) + np.sin(self.angle)*(self.beta_y-self.position[1])
        yp =-np.sin(self.angle)*(self.beta_x-self.position[0]) + np.cos(self.angle)*(self.beta_y-self.position[1])

        # convert to elliptical radius
        r = np.sqrt(self.q*xp**2 + yp**2/self.q)
        
        # Intensity is 1d function of r
        self.intensity = self.Flux*np.exp(-0.5*(r/self.sigma)**2)/(2*np.pi*self.sigma**2)
        return

# ----------------------------------------------------------------------

    def build_from_clumps(self):
        return

# ----------------------------------------------------------------------

    def build_sersic_clumps(self):
        return
    
