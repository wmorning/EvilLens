# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 15:06:23 2015

@author: wmorning
"""
# ===========================================================================

import numpy as np
import evillens as evil
from scipy import interpolate
import subprocess
try:
    import drivecasa
except ImportError:
    print("failed to load drivecasa.  You may not be able to use saboteur")


# ===========================================================================

class Saboteur(object):
    '''
    An object class which sabotages Visibilities data
    '''

    def __init__(self, K, wavelength):
        
        self.path = None
        self.Visibilities = None
        self.antenna1 = None
        self.antenna2 = None
        self.u = None
        self.v = None
        
        self.W = 1000.0
        self.L0 = 6000.0
        
        self.K = K
        self.wavelength = wavelength
        
        self.antennaX = None
        self.antennaY = None
        return
    
# ---------------------------------------------------------------------------
    
    def read_data_from(self, MeasurementSet, antennaconfig):
        '''
        Reads data from a measurement set, and stores visibilties, 
        uv coordinates, and the corresponding antennas.  Also loads in
        position coordinates of antennas from antennaconfig argument.  Should
        be same as the antennaconfig as listed in the measurement set name 
        in order for phase errors to be correct.
        '''
        casa=drivecasa.Casapy()
        
        self.path = str(MeasurementSet)        
        
        script = ['ms.open("%(path)s",nomodify=False)' % {"path": self.path}\
                  , 'recD=ms.getdata(["data"])', 'aD=recD["data"]', 'UVW=ms.getdata(["UVW"])', 'uvpoints=UVW["uvw"]'\
                  ,'u=uvpoints[0]', 'v=uvpoints[1]'\
                  , 'antD1=ms.getdata(["antenna1"])'\
                  , 'antD1=antD1["antenna1"]'\
                  ,'antD2=ms.getdata(["antenna2"])'\
                  , 'antD2=antD2["antenna2"]', 'ms.close()'\
                  ,'print([" "+str(v[i])+" " for i in range(len(v)) ])'\
                  ,'print([" "+str(u[i])+" "for i in range(len(u))])'\
                  ,'print([" "+str(aD[0][i][j])+" "  for i in range(len(aD[0])) for j in range(len(aD[0][0]))])'\
                  ,'print([" "+str(antD1[i])+" " for i in range(len(antD1))])'\
                  ,'print([" "+str(antD2[i])+" " for i in range(len(antD2))])']
        
        data = casa.run_script(script)
              
        
        self.v = np.array((data[0][25].split())[1::3],float)
        self.u = np.array((data[0][28].split())[1::3],float)
        self.Visibilities = np.array(np.array((data[0][31].split())[1::3]),complex)
        self.antenna1 = np.array((data[0][34].split())[1::3],int)
        self.antenna2 = np.array((data[0][37].split())[1::3],int)
        
        #get list of antenna coordinates using location given in casapath
        self.get_antenna_coordinates(antennaconfig)
        
# ---------------------------------------------------------------------------
        
    def get_antenna_coordinates(self, antennaconfig):
        casa = drivecasa.Casapy()        
        
        CASAdir = casa.run_script(['print(" "+os.getenv("CASAPATH").split(" ")[0]+"/data/alma/simmos/")'])[0][0].split()[0]
        antennaparams = np.genfromtxt(str(CASAdir)+str(antennaconfig))
        self.antennaX = antennaparams[:,0]
        self.antennaY = antennaparams[:,1]
        self.antennaZ = antennaparams[:,2]
        
# ---------------------------------------------------------------------------        

    def add_decoherence(self):
        '''
        Add decoherence to visibilities.  Use baseline to determine rms phase 
        error and convolve the visibilities using equation 4 in Carilli and 
        Holdaway.
        '''
        b = np.sqrt(self.u**2+self.v**2)
        
        for i in range(len(b)):
            if b[i] <= self.W:
                self.Visibilities[i] *= np.exp(-((self.K/self.wavelength)*(b[i]/1000)**(5.0/6.0)*(np.pi/180.0))**2/2)
            elif b[i] >self.W and b[i] <= self.L0:
                self.Visibilities[i] *= np.exp(-((self.K/self.wavelength)*(b[i]/1000)**(1.0/3.0)*(np.pi/180.0))**2/2)
            else:
                self.Visibilities[i] *= np.exp(-((self.K/self.wavelength)*(np.pi/180.0))**2/2)
            
        return

# ---------------------------------------------------------------------------
    
    def add_amplitude_errors(self, rms_error):
        '''
        Add amplitude errors to the visibilities.  Each antenna gets gaussian
        random amplitude error centered around 1 with rms equal to input rms. 
        '''
        errors_real = np.random.normal(1.0, rms_error, int(max(self.antenna2)+1))
        errors_imag = np.random.normal(1.0,rms_error,int(max(self.antenna2)+1))
    
        
    
        for i in range(len(self.Visibilities)):
            self.Visibilities[i] = self.Visibilities[i].real*errors_real[self.antenna1[i]]*errors_real[self.antenna2[i]] \
                                    +1j*self.Visibilities[i].imag*errors_imag[self.antenna1[i]]*errors_imag[self.antenna2[i]]
        
        return
# ---------------------------------------------------------------------------
        
    def get_phases(self,v):
        '''
        Create coordinate grid of rms phases, using the phase structure function.
        '''
        self.velocity = v   
        Nbaselines = (len(self.antennaX)*(len(self.antennaX)-1))
        Ntsteps = len(self.Visibilities)//Nbaselines
        maxX = np.max([np.max(abs(self.antennaX))+self.velocity*Ntsteps \
                       ,np.max(abs(self.antennaY))+self.velocity*Ntsteps])//100 *100 +100
        
        x = np.arange(-maxX,maxX+10.0,10.0)
        y = np.arange(-maxX,maxX+10.0,10.0)
        X,Y = np.meshgrid(x,y)
        phases = np.random.normal(0.0,1.0,(len(x),len(y)))
        
        p2 = np.fft.fftshift(np.fft.fft2(phases),axes={0,1})
        FreqX = np.fft.fftshift(np.fft.fftfreq(len(x), x[1]-x[0] ))
        FreqY = np.fft.fftshift(np.fft.fftfreq(len(y), y[1]-y[0] ))
        
        for i in range(len(FreqX)):
            for j in range(len(FreqY)):
                if np.sqrt(FreqX[i]**2+FreqY[j]**2)>1.0/1000.0:
                    p2[i,j] *= (np.pi/180.0)*(self.K/self.wavelength) \
                               *(0.001/np.sqrt(FreqX[i]**2+FreqY[j]**2))**(5.0/6.0)
                elif np.sqrt(FreqX[i]**2+FreqY[j]**2)<=1.0/1000.0 and np.sqrt(FreqX[i]**2+FreqY[j]**2)>1.0/6000.0:
                    p2[i,j] *= (np.pi/180.0)*(self.K/self.wavelength) \
                               *(0.001/np.sqrt(FreqX[i]**2+FreqY[j]**2))**(1.0/3.0)
                else:
                    p2[i,j] *= (np.pi/180.0)*(self.K/self.wavelength)*(6.0)**(1.0/3.0)
                    
        phases = np.fft.ifft2(np.fft.ifftshift(p2,axes={0,1}))
        self.phases=phases.real
        self.phasecoords_x = x
        self.phasecoords_y = y
        self.Nbaselines = Nbaselines
        self.Ntsteps = Ntsteps
        
        
        
# ---------------------------------------------------------------------------    
    def add_phase_errors(self, v):
        '''
        Create coordinate grid of rms phases, using the phase structure function.
        Assign one phase to each antenna, determined using the antenna's position
        on the grid.  All visibilities are shifted by the phases assigned to each
        of their antennas.  This step is performed for several time steps, between
        which the phase screen is translated.  For simplicity, the
        translation will always be in the x-direction
        -v determines the rate at which the phase grid translates.
        '''
        
        self.get_phases(v)
        
        f_interp = interpolate.RectBivariateSpline(self.phasecoords_x,self.phasecoords_y,self.phases,kx=1,ky=1)
        
        self.phase_errors1 = np.zeros(len(self.Visibilities),float)
        self.phase_errors2 = np.zeros(len(self.Visibilities),float)
        for i in range(len(self.phase_errors1)):
            self.phase_errors1[i] = f_interp(self.antennaX[self.antenna1[i]]+self.velocity*(i//self.Ntsteps),self.antennaY[self.antenna1[i]])
            self.phase_errors2[i] = f_interp(self.antennaX[self.antenna2[i]]+self.velocity*(i//self.Ntsteps),self.antennaY[self.antenna2[i]])
        
        self.Visibilities *= np.exp(1j*(self.phase_errors1-self.phase_errors2))   
        
        return
        
# -------------------------------------------------------------------------
    
    def add_noise(self,rms):
        for i in range(len(self.Visibilities)):
            self.Visibilities[i]+= np.random.normal(0.0,rms)+1j*np.random.normal(0.0,rms)
        return
# -------------------------------------------------------------------------
    
    def sabotage_measurement_set(self):
        '''
        Write the corrupted visibilities back to the original measurement set.
        For this copy measurement set into new set 
        /path/measurementset_sabotaged.ms
        '''        
        
        #create location for new ms, and copy old ms to new location
        self.path_new = self.path[:-3]+'_sabotaged.ms'
        command = ['cp','-R', self.path+'/', self.path_new+'/']
        subprocess.call(command)
        
        visibilities_new = [] #new visibilities to be passed to CASA as a list
        for i in range(len(self.Visibilities)):
            visibilities_new.append(self.Visibilities[i])
            
        
        script = ['ms.open("%(path)s",nomodify=False)' % {"path": self.path_new}\
                      ,'recD=ms.getdata(["data"])']
        script.append('vis_new ='+str(visibilities_new))
        script.append("recD['data'][0,0,:]=vis_new")
        script.append("ms.putdata(recD)")
        script.append("ms.close()")
        
        casa=drivecasa.Casapy()
        output = casa.run_script(script)
        print(output)
        
        
        
        return
