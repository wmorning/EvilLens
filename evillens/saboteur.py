# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 15:06:23 2015

@author: wmorning
"""
# ===========================================================================

import numpy as np
import evillens as evil
from scipy import interpolate
import matplotlib.pyplot as plt
import subprocess
import struct
import astropy.convolution as astconv
try:
    import drivecasa
except ImportError:
    print("failed to load drivecasa.  You may not be able to use saboteur")


# ===========================================================================

class Saboteur(object):
    '''
    An object class which sabotages Visibilities data
    '''

    def __init__(self, K, wavelength, integration_time):
        
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
        self.integration_time = integration_time
        
        self.antennaX = None
        self.antennaY = None
        self.phases = None
        return
    
# ---------------------------------------------------------------------------
    
    def read_data_from(self, MeasurementSet, antennaconfig,Blueberry=False):
        '''
        Reads data from a measurement set, and stores visibilties, 
        uv coordinates, and the corresponding antennas.  Also loads in
        position coordinates of antennas from antennaconfig argument.  Should
        be same as the antennaconfig as listed in the measurement set name 
        in order for phase errors to be correct.
        - MeasurementSet is the directory of the data (either *.ms or */)
        - Setting Blueberry flag to True reads binary files written in Blueberry
          format.
        '''
        if Blueberry is False:
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
            
        else:
            with open(MeasurementSet+'v.bin', mode='rb') as file:
                fileContent = file.read()
                self.v = np.array(struct.unpack("d"*(len(fileContent)//8),fileContent))
            file.close()
            
            with open(MeasurementSet+'u.bin', mode='rb') as file:
                fileContent = file.read()
                self.u = np.array(struct.unpack("d"*(len(fileContent)//8),fileContent))
            file.close()
            
            with open(MeasurementSet+'Vis_chan_0.bin', mode='rb') as file:
                fileContent = file.read()
                Visibilities = np.array(struct.unpack("d"*(len(fileContent)//8),fileContent))
                self.Visibilities = Visibilities[::2]+1j*Visibilities[1::2]
            file.close()
            
            with open(MeasurementSet+'ant_1.bin', mode='rb') as file:
                fileContent = file.read()
                self.antenna1 = np.array(struct.unpack("d"*(len(fileContent)//8),fileContent))
            file.close()
            
            with open(MeasurementSet+'ant_2.bin', mode='rb') as file:
                fileContent = file.read()
                self.antenna2 = np.array(struct.unpack("d"*(len(fileContent)//8),fileContent))
            file.close()
        
        #get list of antenna coordinates using location given in casapath
        self.get_antenna_coordinates(antennaconfig)
        
# ---------------------------------------------------------------------------
        
    def ms_to_bin(self,measurementset,filedir,Nchannels=1):
        
        script1 = ['ms.open({0})'.format(measurementset), \
                'recD = ms.getdata(["data"])', 'aD = recD["data"]', \
                'print(len(aD[0]))']
                
        casa = drivecasa.Casapy()
        temp = casa.run_script(script1)
        Nchannels = int(temp[0][7])
                   
        main_script = ['from array import array' , 'import struct', 'import numpy as np', \
                'WritebaseName={0}'.format(str(filedir)),'print(WritebaseName)','ms.open({0})'.format(measurementset), \
                'recD = ms.getdata(["data"])','aD=recD["data"]', \
                'UVW=ms.getdata(["UVW"])','uvpoints=UVW["uvw"]', \
                'u=uvpoints[0]','v=uvpoints[1]', \
                'antD1 = ms.getdata(["antenna1"])','antD1=antD1["antenna1"]', \
                'antD2 = ms.getdata(["antenna2"])','antD2=antD2["antenna2"]', \
                'f = open(WritebaseName + "ant_1.bin","wb")', \
                'data = struct.pack("d"*len(antD1),*antD1)', \
                'f.write(data)','f.close()', \
                'f = open(WritebaseName + "ant_2.bin","wb")', \
                'data = struct.pack("d"*len(antD2),*antD2)', \
                'f.write(data)', 'f.close()', \
                'f = open(WritebaseName + "u.bin", "wb")', \
                'data = struct.pack("d"*len(u),*u)', \
                'f.write(data)','f.close()', \
                'f = open(WritebaseName + "v.bin", "wb")', \
                'data= struct.pack("d"*len(v),*v)', \
                'f.write(data)','f.close()']
                
        for i in range(Nchannels):
            main_script.append('datalist = np.zeros([2*len(aD[0][{0}])],float)'.format(i))
            main_script.append('datalist[::2] = aD[0][{0}][:].real'.format(i))
            main_script.append('datalist[1::2]= aD[0][{0}][:].imag'.format(i))
            main_script.append('f = open(WritebaseName+"Vis_chan_"+str({0})+".bin","wb")'.format(i))
            main_script.append('data = struct.pack("d"*len(datalist),*datalist)')
            main_script.append('f.write(data)')
            main_script.append('f.close()')

                
        
        temp = casa.run_script(main_script)

        print("Done \n")
        return
        
# ---------------------------------------------------------------------------
        
    def get_antenna_coordinates(self, antennaconfig):
        casa = drivecasa.Casapy()        
        
        CASAdir = casa.run_script(['print(" "+os.getenv("CASAPATH").split(" ")[0]+"/data/alma/simmos/")'])[0][0].split()[0]
        antennaparams = np.genfromtxt(str(CASAdir)+str(antennaconfig))
        self.antennaX = antennaparams[:,0]
        self.antennaY = antennaparams[:,1]
        self.antennaZ = antennaparams[:,2]
        
        self.antennaX -= (np.max(self.antennaX)+np.min(self.antennaX))/2.0
        self.antennaY -= (np.max(self.antennaY)+np.min(self.antennaY))/2.0
    
        self.get_Nbaselines()
        self.get_Ntsteps()
        return
        
# ---------------------------------------------------------------------------    
    def get_Nbaselines(self):
        
        if hasattr(self, 'antennaX') is False:
            raise Exception("You need to get an antenna configuration \n")
        self.Nbaselines = (len(self.antennaX)*(len(self.antennaX)-1))/2
        self.Nantennas = len(self.antennaX)
        
# ---------------------------------------------------------------------------
    def get_Ntsteps(self):
        if hasattr(self, 'antennaX') is False:
            raise Exception("You need to load an antenna configuration \n")
        if hasattr(self, 'Visibilities') is False:
            raise Exception("You need to load visibilities \n")
        if hasattr(self, "Nbaselines") is False:
            self.get_Nbaselines()
        self.Ntsteps = len(self.Visibilities)//self.Nbaselines
        
        if len(self.Visibilities) % self.Nbaselines != 0:
            print('WARNING: shape mismatch between visibilities and baselines')
        return
        
# ---------------------------------------------------------------------------        

    def add_decoherence(self , bintime = 5.0, phases = True, cellsize = 10.0, \
                        velocity = 10.0 , randseed = 1 , oversample = False , \
                        oversample_tstep = 0.1):
        '''
        Add decoherence to visibilities.  We do this by creating a high resolution
        phase screen, and sampling the phase at each antenna for small time steps.
        the resulting phase is averaged over segments that are "bintime" long.
        should approximately cause a reduction in amplitude of 
        exp( - 0.5* phi_rms ^ 2 ) to the signal.  The phases flag tells the
        function if we retain just the amplitude reduction, or if we include
        the mean phase shift as a phase error.
        
        This function automatically bins the visibilities.  For small datasets,
        where no binning is required this function samples the phase at a faster
        rate than the input integration time to provide an accurate magnitude
        of the decoherence.
        
        - bintime is given in seconds.  It cannot be less than the integration
        time, and it must return an integer when divided into the total time and 
        by the integration time.  ---> bug fixes later
        - cellsize is size of cells in phase screen
        - velocity is velocity of phase screen in m/s
        - if phases is False, we assume that phase calibration on the binning
        timescale is perfect, and so we only observe an amplitude reduction
        due to decoherence.  If its True, the phases are added before binning
        '''
        #raise Exception("can't add decoherence just yet! \n")
        assert  bintime >= self.integration_time 
        
        if cellsize <12.0:
            convolution = True
        else:
            convolution = False
            
        if oversample == False:
            v = velocity*self.integration_time
            self.assign_phases_to_antennas(v=v,fast=False,cellsize=cellsize,convolution=convolution,randseed=randseed)
            
            if phases == False:
                self.bin_visibilities(bintime)
                
                # calculate the phase errors, and reshape their grid so that we  
                # can average along one axis.
                phaseErrs = (self.phase_errors1-self.phase_errors2 \
                ).reshape([len(self.phase_errors1)/self.Nbaselines/self.Nsteps_binning,  \
                self.Nsteps_binning , self.Nbaselines])
                
                # now calculate the coherence by averaging exp( 1j * phi)
                # and subtract the overall phase error by dividing
                # by exp( 1j * phi_mean ) .
                coherence = (np.mean(np.exp(1j*phaseErrs),axis=1) \
                            / np.exp(1j*np.mean(phaseErrs,axis=1))).reshape( \
                            self.Ntsteps*self.Nbaselines/self.Nsteps_binning)
                
                self.Visibilities *= coherence
            
            else:
                
                # add phase errors, and then bin them.  Decoherence comes 
                # in naturally this way
                self.Visibilities *= np.exp(1j*(self.phase_errors1-self.phase_errors2))
                self.bin_visibilities(bintime)
        else:
            raise Exception("can't do oversampled phase grid yet \n")
                

                
#        for i in range(len(self.Visibilities)):
#            b = np.sqrt((self.antennaX[self.antenna1[i]]-self.antennaX[self.antenna2[i]])**2 \
#                +(self.antennaY[self.antenna1[i]]-self.antennaY[self.antenna2[i]])**2)
#            
#            if b <= self.W:
#                self.Visibilities[i] *= np.exp(-((self.K/(1000.0*self.wavelength))*(b/1000.0)**(5.0/6.0)*(np.pi/180.0))**2/2)
#            elif b >self.W and b <= self.L0:
#                self.Visibilities[i] *= np.exp(-((self.K/(1000.0*self.wavelength))*(b/1000.0)**(1.0/3.0)*(np.pi/180.0))**2/2)
#            else:
#                self.Visibilities[i] *= np.exp(-((self.K/(1000.0*self.wavelength))*(np.pi/180.0)*(6.0**(1.0/3.0)))**2/2)
            
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
        
    def get_phases(self, v , fast=False , cellsize = 10.0 ,convolution=False,randseed=1):
        '''
        Create coordinate grid of rms phases, using the phase structure function.
        '''
        self.velocity = v   
        Nbaselines = (len(self.antennaX)*(len(self.antennaX)-1))/2
        Ntsteps = len(self.Visibilities)//Nbaselines
        self.cellsize = cellsize
        #determine size of the grid        
        maxX = (np.max(self.antennaX) + (np.max(self.antennaX)-np.min(self.antennaX))\
               +self.velocity*Ntsteps) //100 *100 +100
        minX = np.min(self.antennaX) //100 *100 -100
        maxY = 2 * (np.max(self.antennaY) //100 *100+100)
        minY = 2 * (np.min(self.antennaY) //100 *100-100)        
        
        x = np.arange(minX,maxX+cellsize,cellsize)
        y = np.arange(minY,maxY+cellsize,cellsize)
        
        #generate pseudo random numbers
        np.random.seed(randseed)
        phases = np.random.normal(0.0,1.0,(len(y),len(x))) 
        
        p2 = np.fft.fft2(phases) /self.cellsize
        FreqX = np.fft.fftfreq(len(x), 1.0/float(len(x)) )*2.0*np.pi/(x[-1]-x[0])
        FreqY = np.fft.fftfreq(len(y), 1.0/float(len(y))) *2.0*np.pi/(y[-1]-y[0])
        if fast==False:
            for i in range(len(FreqX)):
                for j in range(len(FreqY)):
                    if np.sqrt(FreqX[i]**2+FreqY[j]**2)>1.0/1000.0:
                        p2[j,i] *= (np.pi/180.0)*(self.K/self.wavelength) \
                                *np.sqrt(0.0365)*(1000.0*np.sqrt(FreqX[i]**2 \
                                +FreqY[j]**2))**(-11.0/6.0)
                    elif np.sqrt(FreqX[i]**2+FreqY[j]**2)<=1.0/1000.0 and np.sqrt(FreqX[i]**2+FreqY[j]**2)>1.0/6000.0:
                        p2[j,i] *= (np.pi/180.0)*(self.K/self.wavelength) \
                                *np.sqrt(0.0365)*(1000.0*np.sqrt(FreqX[i]**2\
                                +FreqY[j]**2))**(-5.0/6.0)
                    else:
                        #p2[j,i] *= 0*(np.pi/180.0)*np.sqrt(0.0365) \
                        #        *(self.K/self.wavelength)*(6.0)**(5.0/6.0)
                        p2[j,i] *= (np.pi/180.0)*np.sqrt(0.0365) \
                                *(self.K/self.wavelength)*(6.0)**(-5.0/6.0)
                    
        #  Faster way;  uses slicing and np.where rather than double for loop
        else:       
            FreqX,FreqY = np.meshgrid(FreqX,FreqY)
            p2[np.where(np.sqrt(FreqX**2+FreqY**2)>=1.0/1000.0)] *= (np.pi/180.0)\
            *(self.K/self.wavelength)*np.sqrt(0.0365)*(1000.0 \
            *np.sqrt(FreqX[np.where(np.sqrt(FreqX**2+FreqY**2)>=1.0/1000.0)]**2 \
            +FreqY[np.where(np.sqrt(FreqX**2+FreqY**2)>=1.0/1000.0)]**2))**(-11.0/6.0)
            p2[np.where((np.sqrt(FreqX**2+FreqY**2)<1.0/1000.0) & \
            (np.sqrt(FreqX**2+FreqY**2) >=1.0/6000.0)) ] *= (np.pi/180.0) \
            *np.sqrt(0.0365)*(1000.0*np.sqrt(FreqX[np.where((np.sqrt(FreqX**2 \
            +FreqY**2)<1.0/1000.0) & (np.sqrt(FreqX**2+FreqY**2) >=1.0/6000.0))]**2 \
            + FreqY[np.where((np.sqrt(FreqX**2+FreqY**2)<1.0/1000.0) & \
            (np.sqrt(FreqX**2+FreqY**2) >=1.0/6000.0))]**2))**(-5.0/6.0)
            p2[np.where(np.sqrt(FreqX**2+FreqY**2) < 1.0/6000.0) ] *= (np.pi/180.0)\
            *np.sqrt(0.0365)*(self.K/self.wavelength)*(6.0)**(-5.0/6.0) 
        
        phases = np.fft.ifft2(p2)
        if convolution ==True:        
            # convolve this with tophat kernel of ALMA antenna size (12m)
            tophat_kernel = astconv.Tophat2DKernel(12//self.cellsize)
            self.phases= astconv.convolve(4.0*np.pi*phases.real, tophat_kernel, boundary='wrap',normalize_kernel=True)
        else:
            self.phases = 4.0*np.pi*phases.real
        self.phasecoords_x = x
        self.phasecoords_y = y
        self.Nbaselines = Nbaselines
        self.Ntsteps = Ntsteps
        
# ---------------------------------------------------------------------------
        
    def assign_phases_to_antennas(self , v , fast = False , cellsize = 10.0 , \
                                    convolution = False , randseed = 1 ):
        if self.phases is None or v != self.velocity:                                
            self.get_phases(v,fast,cellsize,convolution,randseed)
        
        f_interp = interpolate.RectBivariateSpline(self.phasecoords_y,self.phasecoords_x,self.phases,kx=1,ky=1)
        
        self.phase_errors1 = np.zeros(len(self.Visibilities),float)
        self.phase_errors2 = np.zeros(len(self.Visibilities),float)
        for i in range(len(self.phase_errors1)):
            self.phase_errors1[i] = f_interp(self.antennaY[self.antenna1[i]],self.antennaX[self.antenna1[i]]+self.velocity*(i//self.Nbaselines))
            self.phase_errors2[i] = f_interp(self.antennaY[self.antenna2[i]],self.antennaX[self.antenna2[i]]+self.velocity*(i//self.Nbaselines))
        
        self.antennaphases = np.zeros([len(self.antennaX),self.Ntsteps], float)
        for i in range(self.Ntsteps):
            for j in range(len(self.antennaX)):
                self.antennaphases[j,i] = f_interp(self.antennaY[j],self.antennaX[j]+self.velocity*i)        
        
# ---------------------------------------------------------------------------    
    def add_phase_errors(self, v , fast = False, cellsize = 10.0 ,convolution=False, randseed = 1):
        '''
        Create coordinate grid of rms phases, using the phase structure function.
        Assign one phase to each antenna, determined using the antenna's position
        on the grid.  All visibilities are shifted by the phases assigned to each
        of their antennas.  This step is performed for several time steps, between
        which the phase screen is translated.  For simplicity, the
        translation will always be in the x-direction
        -v determines the rate at which the phase grid translates.
        -cellsize is size of phase cells (in meters).
        -convolution flags whether a user wants to convolve the phase screen with
            the 12m size of ALMA antennas.
        -randseed allows the user to specify the input random number seed in order
            to control the phase errors.  Used mostly for testing purposes.
        '''
#        if self.phases is None or v !=self.velocity:            
#            self.get_phases(v, fast, cellsize,convolution,randseed)
#        
#        
#        f_interp = interpolate.RectBivariateSpline(self.phasecoords_y,self.phasecoords_x,self.phases,kx=1,ky=1)
#        
#        self.phase_errors1 = np.zeros(len(self.Visibilities),float)
#        self.phase_errors2 = np.zeros(len(self.Visibilities),float)
#        for i in range(len(self.phase_errors1)):
#            self.phase_errors1[i] = f_interp(self.antennaY[self.antenna1[i]],self.antennaX[self.antenna1[i]]+self.velocity*(i//self.Nbaselines))
#            self.phase_errors2[i] = f_interp(self.antennaY[self.antenna2[i]],self.antennaX[self.antenna2[i]]+self.velocity*(i//self.Nbaselines))
#        
#        self.antennaphases = np.zeros([len(self.antennaX),self.Ntsteps], float)
#        for i in range(self.Ntsteps):
#            for j in range(len(self.antennaX)):
#                self.antennaphases[j,i] = f_interp(self.antennaY[j],self.antennaX[j]+self.velocity*i)
        
        self.assign_phases_to_antennas( v, fast, cellsize, convolution, randseed)
        
        self.Visibilities *= np.exp(1j*(self.phase_errors1-self.phase_errors2))   
        
        return

# -------------------------------------------------------------------------

    def bin_visibilities(self , bintime = 5.0):
        '''
        Bin the visibilities in time intervals of bintime.  If phase errors
        are added, then this will cause decorrelation of the visibilities.
        
        For now, we require that the binning time divides evenly into the
        total time, and that the integration time divides evenly into the
        binning time
        '''
        #Nsteps must be an integer.
        if self.integration_time < 1:
            bintime_ms = int(bintime*1000)
            integration_ms = int(self.integration_time*1000)
            Nsteps = bintime_ms // integration_ms
            assert bintime_ms % integration_ms == 0
            assert self.Ntsteps % Nsteps == 0
        else:
            Nsteps = int(bintime // self.integration_time)
            assert(abs(int(bintime) % int(self.integration_time))/float(self.integration_time) < 10**-4)        
            assert self.Ntsteps % Nsteps == 0
        
        # reshape data to prepare for averaging
        self.Visibilities = self.Visibilities.reshape([len(self.Visibilities)/self.Nbaselines/Nsteps,Nsteps,self.Nbaselines])
        self.u = self.u.reshape([len(self.u)/self.Nbaselines/Nsteps,Nsteps,self.Nbaselines])
        self.v = self.v.reshape([len(self.v)/self.Nbaselines/Nsteps,Nsteps,self.Nbaselines])
        self.antenna1 = self.antenna1.reshape([len(self.antenna1)/self.Nbaselines/Nsteps,Nsteps,self.Nbaselines])        
        self.antenna2 = self.antenna2.reshape([len(self.antenna2)/self.Nbaselines/Nsteps,Nsteps,self.Nbaselines])        
                
        # average u,v,visibilities along the first axis, and return to original
        # shape
        self.Visibilities = np.mean(self.Visibilities,axis=1).reshape(self.Nbaselines*self.Ntsteps/Nsteps)
        self.u = np.mean(self.u,axis=1).reshape(self.Nbaselines*self.Ntsteps/Nsteps)
        self.v = np.mean(self.v,axis=1).reshape(self.Nbaselines*self.Ntsteps/Nsteps)
        self.antenna1 = self.antenna1[:,0,:].reshape(self.Nbaselines*self.Ntsteps/Nsteps)
        self.antenna2 = self.antenna2[:,0,:].reshape(self.Nbaselines*self.Ntsteps/Nsteps)

        # set new integration time
        self.integration_time = bintime
        self.Nsteps_binning = Nsteps
# -------------------------------------------------------------------------
    
    def add_noise(self,rms,seed=1):
        self.noise_rms = rms
        np.random.seed(seed)
        
        self.noise = np.random.normal(0.0,rms,len(self.Visibilities))+1j*np.random.normal(0.0,rms,len(self.Visibilities))
        self.Visibilities += self.noise

#        for i in range(len(self.Visibilities)):
#            self.Visibilities[i]+= np.random.normal(0.0,rms)+1j*np.random.normal(0.0,rms)
        return

# -------------------------------------------------------------------------

    def write_phase_matrices(self,datadir,Numphaseintervals):
        '''
        Create matrix files Rowisone.bin, Colisone.bin, Rowisminusone.bin
        and Colisminusone.bin which contain index labels for the +1 and -1 
        values in the dtheta/dphi matrix.  Assumes that the phase matrix
        is a Nvisibilities by Numphaseintervals matrix, and that the 
        visibilities are listed in the standard CASA simobserve format.
            
        Right now it breaks the observation into approximately equal length
        chunks (the last one may be slightly shorter).  In the future, we 
        could upgrade by specifying the intervals where the phase is expected
        to change.
    
        We'll also specify that the phase of the zeroth antenna is our 
        reference phase (so we set it to 0).
        '''
    
        # Total number of integrations may have changed due to binning.
        Nintegrations = len(self.Visibilities) // self.Nbaselines
        assert Nintegrations >= Numphaseintervals
        IntperInterval = Nintegrations // Numphaseintervals
        # in case of uneven division, add an additional integration to each 
        # interval (last one becomes shorter)        
        if Nintegrations % Numphaseintervals != 0:
            IntperInterval +=1 
        
        # number of visibility points in each phase interval
        ObsperInterval = IntperInterval * self.Nbaselines
        
    
        rowisone = np.zeros(len(self.Visibilities))
        colisone = np.zeros(len(self.Visibilities))
        rowisminusone = np.zeros(len(self.Visibilities))
        colisminusone = np.zeros(len(self.Visibilities))
    
        for i in range(len(rowisone)):
            if self.antenna1[i] ==0: #exclude antenna 0
                rowisminusone[i] = i
                colisminusone[i] = self.antenna2[i]-1+(i//ObsperInterval)*(self.Nantennas-1)
            else:
                rowisone[i] = i
                colisone[i] = self.antenna1[i]-1 +(i//ObsperInterval)*(self.Nantennas-1)
                rowisminusone[i] = i
                colisminusone[i] = self.antenna2[i]-1+(i//ObsperInterval)*(self.Nantennas-1)
        
        # clip where antenna1 is present from this list
        colisone = colisone[(rowisone !=0)]
        rowisone = rowisone[(rowisone !=0)]
        
        # have the matrices we want, now write to data.
        f = open(str(datadir)+'ROWisone.bin','wb')
        data = struct.pack('i'*len(rowisone),*rowisone)
        f.write(data)
        f.close()
        
        g = open(str(datadir)+'COLisone.bin','wb')
        data = struct.pack('i'*len(colisone),*colisone)
        g.write(data)
        g.close()
        
        h = open(str(datadir)+'ROWisminusone.bin','wb')
        data = struct.pack('i'*len(rowisminusone),*rowisminusone)
        h.write(data)
        h.close()
        
        i = open(str(datadir)+'COLisminusone.bin','wb')
        data = struct.pack('i'*len(colisminusone),*colisminusone)
        i.write(data)
        i.close()
    
        return
    
# -------------------------------------------------------------------------
    
    def sabotage_measurement_set(self,lenstool=False):
        '''
        Write the corrupted visibilities back to the original measurement set.
        For this copy measurement set into new set 
        /path/measurementset_sabotaged.ms
        
        Alternative method for large data sets.  Write visibilities to file 
        individually.  Make new directory for these with name 
        /path/measurement_sabotaged/.        
        '''        
        if lenstool == True:
            '''
            Place data in format for use with lens tool code.
            that means uv data in wavelengths, visibilties 
            and sigma squared inverse in single column vectors,
            and all data saved as doubles using struct.pack
            '''
            
            u = self.u / self.wavelength
            v = self.v / self.wavelength
            sigma_squared_inv = 1.0/self.noise_rms**2 *np.ones(2*len(self.Visibilities),float)
            vis = np.empty(2*len(self.Visibilities),float)
            vis[0::2] = self.Visibilities.real
            vis[1::2] = self.Visibilities.imag
            
            # Can only do single channel data now.
            chan = np.zeros(len(self.u),float)
            
            self.path_new = self.path[:-3]+'_sabotaged/'
            command = ['mkdir' , self.path_new]
            subprocess.call(command)
            
            f = open(self.path_new+'u.bin','wb')
            data = struct.pack('d'*len(u),*u)
            f.write(data)
            f.close()
            g = open(self.path_new+'v.bin','wb')
            data = struct.pack('d'*len(v),*v)
            g.write(data)
            g.close()
            h = open(self.path_new+'chan.bin','wb')
            data = struct.pack('d'*len(chan),*chan)
            h.write(data)
            h.close()
            i = open(self.path_new+'vis_chan_0.bin','wb')
            data = struct.pack('d'*len(vis),*vis)
            i.write(data)
            i.close()
            j = open(self.path_new+'sigma_squared_inv.bin','wb')
            data = struct.pack('d'*len(sigma_squared_inv),*sigma_squared_inv)
            j.write(data)
            j.close()
            
        else:
        
            if len(self.Visibilities)<10**6:
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
        
            else:
                self.path_new = self.path[:-3]+'_sabotaged/'
                command = ['mkdir', self.path_new]
                subprocess.call(command)
            
                f = open(self.path_new+'Vis_real.txt', 'w')
                for i in range(len(self.Visibilities)):
                    f.write(str(self.Visibilities[i].real))
                    f.write('\n')
                    f.close()
            
                f = open(self.path_new+'Vis_imag.txt', 'w')
                for i in range(len(self.Visibilities)):
                    f.write(str(self.Visibilities[i].imag))
                    f.write('\n')
                    f.close()
            
            
                
        
        return
# -------------------------------------------------------------------------
        
    def plot(self,mapname,Figsize=[10,10]):
        
        if mapname == 'structure function':
            #first get rms phases vs distance
            dist = []
            rmsPhase = []            
            for i in range(self.Nbaselines):
                dist.append(np.sqrt((self.antennaX[self.antenna1[i]]-self.antennaX[self.antenna2[i]])**2 \
                    +(self.antennaY[self.antenna1[i]]-self.antennaY[self.antenna2[i]])**2))
                rmsPhase.append(np.sqrt(np.mean((self.phase_errors1[i::self.Nbaselines] \
                    -self.phase_errors2[i::self.Nbaselines])**2))*180/np.pi)
            
            self.dist = np.array(dist)
            self.rmsPhase = np.array(rmsPhase )           
            
            plt.figure(figsize=Figsize)
            
            xpoints = np.arange(10,20000,20.0)
            predictions = (1.0/((1.0/(1.0/(((1.0/(self.K/(1000*self.wavelength)*((xpoints/1000.0)**(5.0/6.0))))**(40.0) \
                +(1.0/(self.K/(1000.0*self.wavelength)*(xpoints/1000.0)**(1.0/3.0)))**(40.0)))**(1.0/40.0)))**(40.0) \
                +(1.0/(self.K/(1000.0*self.wavelength)*6.0**(1.0/3.0)+0*xpoints))**(40.0))**(1.0/40.0))
            
            plt.plot(xpoints,predictions,'b-')
            plt.plot(dist,rmsPhase,'k.')
            plt.plot([1000,1000],[0,400], 'r-')
            plt.plot([(6000),(6000)],[0,400], 'r-')
            plt.xlim(np.min(dist)-10.0,np.max(dist)+100.0)
            plt.ylim(np.min(rmsPhase)/1.1,np.max(rmsPhase)+5.0)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('baseline (m)')
            plt.ylabel('rms phase (deg)')
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
            
        return

# -------------------------------------------------------------------------