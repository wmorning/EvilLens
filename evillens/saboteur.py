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
import os
from scipy.interpolate import interp1d
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

    def Simulate_observation(self,lens,u,v,ant1,ant2,antennaconfig):
        '''
        Takes in a lens object, as well as uv configuration files 
        (with u and v in meters) and the name of the antenna configuration
        file used to generate the u and v files.
        
        Creates the visibilities, and attaches everything to the correct
        location in the object.  Meant to simulate the read_data_from 
        function but without having to use CASA.
        
        
        TO DO: Build the u and v list from the antenna configuration file. 
        This will require some edits to the get_antenna_coordinates
        function that will disturb other functionality.  Leave as it is for 
        now.
        '''
        
        assert issubclass(type(lens),evil.GravitationalLens)
        
        if (type(u) == str):
            u = evil.load_binary(u)
            u /= self.wavelength
        if (type(v) == str):
            v = evil.load_binary(v)
            v /= self.wavelength
        if (type(ant1) == str):
            ant1 = evil.load_binary(ant1)
        if (type(ant2) == str):
            ant2 = evil.load_binary(ant2)
            
        self.u = u
        self.v = v
        self.antenna1 = np.rint(ant1).astype(int)
        self.antenna2 = np.rint(ant2).astype(int)
        
        x = lens.image_x / 3600. / 180. * np.pi
        y = lens.image_y / 3600. / 180. * np.pi
        
        vis = np.zeros(len(u),'complex')
        
        for i in range(len(u)):
            vis[i] = np.sum(lens.image * np.exp( -2j*np.pi *(x*self.u[i] +y*self.v[i])))
        
        self.Visibilities = vis
        
        self.get_antenna_coordinates(antennaconfig)
        
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
            
        # convert antenna IDs to integer for indexing
        self.antenna1 = np.rint(self.antenna1).astype(int)
        self.antenna2 = np.rint(self.antenna2).astype(int)
        
        #get list of antenna coordinates using location given in casapath
        self.get_antenna_coordinates(antennaconfig)

# ---------------------------------------------------------------------------
        
    def ms_to_bin(self,measurementset,filedir):
        '''
        Opens a CASA measurement set (using the drivecasa software) and writes
        the visibilities, uv coordinates, antennas, sigmas, and times of all
        the integrations to binary files (binary files are easier to deal with
        outside of CASA).  This should be regarded as the first step in the 
        process of reducing the data, as the sigmas should still be scaled, and
        the dOdphase matrices should still be built.
        
        Takes:
        
        measurementset - The name of the CASA measurement set to be converted
        
        filedir        - The location to which the files will be written.
        
        Returns:
        
        Nchannels      - The number of channels in the measurement set
        
        Nspw           - The number of spectral windows in the ms.
        '''
        
        # CASA script to get number of SPWs and channels.
        script1 = ['ms.open({0})'.format(measurementset), \
                'recD = ms.getdata(["axis_info"])', 'aD = recD["axis_info"]["freq_axis"]["chan_freq"]', \
                'print(len(aD))','print(len(aD[0]))']
                
        casa = drivecasa.Casapy()
        temp = casa.run_script(script1)
        
        # Fool proof get spectral windows and channels.
        Nchanknown = False
        for i in range(len(temp[0])):
            if Nchanknown is False:
                try:
                    Nchannels = int(temp[0][i])
                    Nchanknown = True
                except:
                    continue
            else:
                try:
                    Nspw = int(temp[0][i])
                else:
                    continue
        
        
        # CASA script to write all data to binary.           
        main_script = ['from array import array' , 'import struct', 'import numpy as np', \
                'WritebaseName={0}'.format(str(filedir)),'print(WritebaseName)','ms.open({0})'.format(measurementset), \
                'recD = ms.getdata(["data","axis_info"])','aD=recD["data"]', \
                'UVW=ms.getdata(["UVW"])','uvpoints=UVW["uvw"]', \
                'freq = recD["axis_info"]["freq_axis"]["chan_freq"]', \
                'u=uvpoints[0]','v=uvpoints[1]', \
                'antD1 = ms.getdata(["antenna1"])','antD1=antD1["antenna1"]', \
                'antD2 = ms.getdata(["antenna2"])','antD2=antD2["antenna2"]', \
                'Sigma = ms.getdata(["sigma"])["sigma"]',\
                'f = open(WritebaseName + "ant_1.bin","wb")', \
                'data = struct.pack("d"*len(antD1),*antD1)', \
                'f.write(data)','f.close()', \
                'f = open(WritebaseName + "ant_2.bin","wb")', \
                'data = struct.pack("d"*len(antD2),*antD2)', \
                'f.write(data)', 'f.close()', \
                'time = ms.getdata(["time"])["time"]',\
                'f=open(WritebaseName + "time.bin","wb")',\
                'data = struct.pack("d"*len(time),*time)',\
                'f.write(data)','f.close()',\
                'print np.sqrt(np.max((aD[0][0][:].real+aD[1][0][:].real)**2+(aD[0][0][:].imag+aD[1][0][:].imag)**2))']
                
        for i in range(Nchannels):
            for j in range(Nspw):
                main_script.append('datalist = np.zeros([2*len(aD[0][{0}])],float)'.format(i))
                main_script.append('datalist[::2] = (aD[0][{0}][:].real+aD[1][{0}][:].real)/2.0'.format(i))
                main_script.append('datalist[1::2]= (aD[0][{0}][:].imag+aD[1][{0}][:].imag)/2.0'.format(i))
                main_script.append('f = open(WritebaseName+"Vis_spw_{0}_chan_{1}.bin","wb")'.format(j,i))
                main_script.append('data = struct.pack("d"*len(datalist),*datalist)')
                main_script.append('f.write(data)')
                main_script.append('f.close()')
                main_script.append('sigmalist = np.zeros([len(Sigma[0])],float)')
                main_script.append('sigmalist[:] = np.sqrt(Sigma[0]**2+Sigma[1]**2)')
                main_script.append('f = open(WritebaseName+"sigma_spw_{0}_chan_{1}.bin","wb")'.format(j,i))
                main_script.append('data = struct.pack("d"*len(sigmalist),*sigmalist)')
                main_script.append('f.write(data)')
                main_script.append('f.close()')
                main_script.append('f = open(WritebaseName + "u_spw_{0}_chan_{1}.bin", "wb")'.format(j,i))
                main_script.append('udata = u/(3*(10**8)/freq[{0}][{1}])'.format(i,j))
                main_script.append('data = struct.pack("d"*len(udata),*udata)')
                main_script.append('f.write(data)')
                main_script.append('f.close()')
                main_script.append('f = open(WritebaseName + "v_spw_{0}_chan_{1}.bin", "wb")'.format(j,i))
                main_script.append('vdata = v/(3*(10**8)/freq[{0}][{1}])'.format(i,j))
                main_script.append('data= struct.pack("d"*len(v),*vdata)')
                main_script.append('f.write(data)')
                main_script.append('f.close()')
                main_script.append('np.savetxt(WritebaseName+"chan_wav.txt",3*10**8/freq)')

                
        
        temp = casa.run_script(main_script)
        
        print temp  # Parse for errors.
        print("Measurement set converted to binary.  Data written to:  {0} \n".format(filedir))
        return Nchannels , Nspw
        
# ---------------------------------------------------------------------------

    def get_sigma_scaling(self,u,v,vis,sigma):
        '''
        Given a set of data with unscaled errors, determine the scaling
        necessary to make the errors indicative of the data.
        
        Takes:
        
        - u (in meters):  The u coordinates of the data
        
        - v (in meters):  The v coordinates of the data
        
        - vis (in mJy):  The visibility data (in complex array format)
        
        - sigma (Arbitrary):  The unscaled noise expectation for each
        visibility (real array format).
        
        Returns:
        
        SIGMA_SCALING:  The number to scale the noise to the 
        correct level.
        
        '''
        
        UVMAX = np.max([abs(u),abs(v)])*1.01

        Nbins = 256
        
        # Count number of visibilities in each bin.
        Bins = np.arange(-UVMAX,UVMAX+12,12.0)
        P,reject1,reject2 = np.histogram2d(u,v,bins=Bins)
        
#        B1 = interp1d(Bins,range(Nbins),'nearest')
#        iB1 = B1(u)
#        iB2 = B1(v)
        
        # Keep only bins with visiblities in them.
        [row,col] = np.where(P!=0)

        # Keep track of noise estimation stats
        NumSkippedBins = 0
        total=0 
        
        # Real and imaginary noise arrays.
        noise_r = np.zeros(u.shape)
        noise_i = np.zeros(v.shape)
        
        indI = np.zeros(u.shape,int)

        for icount in range(len(row)):
            inds = np.where((v>=Bins[col[icount]])&(v<Bins[col[icount]+1])&(u>=Bins[row[icount]])&(u<Bins[row[icount]+1]))[0]
            
            if len(inds) ==1:
                NumSkippedBins += 1
                continue
            elif len(inds)%2 ==1:
                inds = inds[:-1]
            I = inds[::2]
            J = inds[1::2]
            noise_r[I] = (vis[I]-vis[J]).real
            noise_i[I] = (vis[I]-vis[J]).imag
            indI[I] = J
            total += len(inds)
        
        iI = np.where(noise_r != 0)[0]
        
        N = len(iI)
        SIGMA_SCALING = np.sqrt(N/np.sum((noise_r[iI]/np.sqrt(sigma[iI]**2+sigma[indI[iI]]**2))**2)) 
        
        print "Total number of visibilities used was", total 
        print NumSkippedBins , "bins out of" , len(row), "had only one visibility and were skipped."
        
        return SIGMA_SCALING 
        

# ---------------------------------------------------------------------------

    def build_dOdphase(self, ant1, ant2,time,NUM_TIME_STEPS=1):
        
        '''
        Create matrix files Rowisone.bin, Colisone.bin, Rowisminusone.bin
        and Colisminusone.bin which contain index labels for the +1 and -1 
        values in the dtheta/dphi matrix.  Takes 2 lists of antenna numbers
        ant1 and ant2 (contained in a measurement set).
            
        Right now it breaks the observation into approximately equal length
        chunks (the last one may be slightly shorter).  In the future, we 
        could upgrade by specifying the intervals where the phase is expected
        to change.
    
        We'll also specify that the phase of the zeroth antenna is our 
        reference phase (so we set it to 0).
        '''
    
        NUM_ANT = int(np.max([np.max(ant1),np.max(ant2)]))
        NUM_BL  = (NUM_ANT*(NUM_ANT-1))/2
        NUM_TIME_STEPS = int(NUM_TIME_STEPS)
        
        # Determine time cuts (-1 and +1 to guarantee # of intervals)
        tstart  = np.min(time)-1.0
        tend    = np.max(time)+1.0
        tdiff   = tend-tstart
        tswitch = tdiff/float(NUM_TIME_STEPS)

        rowisone      = np.zeros(len(ant1),int)
        colisone      = np.zeros(len(ant1),int)
        rowisminusone = np.zeros(len(ant2),int)
        colisminusone = np.zeros(len(ant2),int)
        
        for i in range(len(rowisone)):
            if ant1[i] == 0: # First antenna phase is always fixed to zero
                rowisminusone[i] = i
                colisminusone[i] = ant2[i]-1 + int((time[i]-tstart)//tswitch)*(NUM_ANT)
            else:
                rowisone[i]      = i 
                colisone[i]      = ant1[i]-1 + int((time[i]-tstart)//tswitch)*(NUM_ANT)
                rowisminusone[i] = i 
                colisminusone[i] = ant2[i]-1 + int((time[i]-tstart)//tswitch)*(NUM_ANT)
                
        # clip where antenna1 is present from this list
        colisone = colisone[(rowisone !=0)]
        rowisone = rowisone[(rowisone !=0)]        
        
        
        return rowisone , colisone , rowisminusone , colisminusone


# --------------------------------------------------------------------------- 
    
    def Reduce_ms(self , MSNAME , OUTPUTDIR ,NUM_TIME_STEPS=1):
        '''
        Last step in data reduction pipeline
        
        Open a ready to go measurement set (binned visibilities, flagged visibilities
        removed, etc.), and write the files that are used by the pipeline to the directory
        OUTPUTDIR. 
        '''
        
        # Create the destination if it does not already exist.
        if not os.path.isdir(OUTPUTDIR):
            os.mkdir(OUTPUTDIR)
        
        
        # Create the temporary directory for exerything.
        print "Performing first conversion to binary."
        Nchan, Nspw = self.ms_to_bin(MSNAME, '"/Users/wmorning/research/data/temp/"')
        
        Vis , ssqinv , u , v , rowisone , colisone , rowisminusone , colisminusone , chan  = \
                self.prepare_data('/Users/wmorning/research/data/temp/',Nspw,Nchan,NUM_TIME_STEPS=NUM_TIME_STEPS)
                
        with open(OUTPUTDIR+'vis_chan_0.bin','wb') as file:
            data = struct.pack('d'*len(Vis),*Vis)
            file.write(data)
        file.close()
        
        with open(OUTPUTDIR+'sigma_squared_inv.bin','wb') as file:
            data = struct.pack('d'*len(ssqinv),*ssqinv)
            file.write(data)
        file.close()
        
        with open(OUTPUTDIR+'u.bin','wb') as file:
            data = struct.pack('d'*len(u),*u)
            file.write(data)
        file.close()
        
        with open(OUTPUTDIR+'v.bin','wb') as file:
            data = struct.pack('d'*len(v),*v)
            file.write(data)
        file.close()
        
        with open(OUTPUTDIR+'chan.bin','wb') as file:
            data = struct.pack('d'*len(chan),*chan)
            file.write(data)
        file.close()
        
        with open(OUTPUTDIR+'ROWisone.bin','wb') as file:
            data = struct.pack('d'*len(rowisone),*rowisone)
            file.write(data)
        file.close()
        
        with open(OUTPUTDIR+'COLisone.bin','wb') as file:
            data = struct.pack('d'*len(colisone),*colisone)
            file.write(data)
        file.close()
        
        with open(OUTPUTDIR+'ROWisminusone.bin','wb') as file:
            data = struct.pack('d'*len(rowisminusone),*rowisminusone)
            file.write(data)
        file.close()
        
        with open(OUTPUTDIR+'COLisminusone.bin','wb') as file:
            data = struct.pack('d'*len(colisminusone),*colisminusone)
            file.write(data)
        file.close()

        print 'Data written to {0}'.format(OUTPUTDIR)
        
        return
        
# ----------------------------------------------------------------------------
    def prepare_data(self, direct,Nspw,Nchan, bintime=None,NUM_TIME_STEPS=1):
        
        wav = np.loadtxt(direct+'chan_wav.txt')
        if len(wav.shape)==0:
            wav = np.array([[wav]])  # bug fix for single spw, single channel case
        elif len(wav.shape)==1:
            wav = np.array([wav])    # bug fix for single spw case
        
        print("Loading data located at:  {0}".format(direct))
        #  load all of the data (all channels, all spws)
        for i in range(Nspw):
            for j in range(Nchan):
                if (i==0) & (j==0):
                    ut = evil.load_binary(direct+'u_spw_0_chan_0.bin')
                    vt = evil.load_binary(direct+'v_spw_0_chan_0.bin')
                    vist = evil.load_binary(direct+'vis_spw_0_chan_0.bin')
                    sigmat = evil.load_binary(direct+'sigma_spw_0_chan_0.bin')
                    ant1t = np.rint(evil.load_binary(direct+'ant_1.bin')).astype(int)
                    ant2t = np.rint(evil.load_binary(direct+'ant_2.bin')).astype(int)
                    timet = evil.load_binary(direct+'time.bin')
                    u     = np.zeros([Nspw,Nchan,   len(ut)   ], float )
                    v     = np.zeros([Nspw,Nchan,   len(vt)   ], float )
                    vis   = np.zeros([Nspw,Nchan, len(vist)/2 ],complex)
                    sigma = np.zeros([Nspw,Nchan, len(sigmat) ], float )
                    ant1  = np.zeros([Nspw,Nchan,   len(ut)   ],  int  )
                    ant2  = np.zeros([Nspw,Nchan,   len(ut)   ],  int  )
                    chan  = np.zeros([Nspw,Nchan,   len(ut)   ],  int  ) 
                    time  = np.zeros([Nspw,Nchan,   len(ut)   ], float )
                    
                    u[i,j,:]     = ut*wav[i,j]
                    v[i,j,:]     = vt*wav[i,j]
                    vis[i,j,:]   = vist[::2]+1j*vist[1::2]
                    sigma[i,j,:] = sigmat
                    ant1[i,j,:]  = ant1t
                    ant2[i,j,:]  = ant2t
                    time[i,j,:]  = timet
                    chan[i,j,:]  +=j+Nchan*i
                    ut = 0
                    vt = 0
                    vist = 0
                    sigmat = 0
                    ant1t = 0
                    ant2t = 0
                    sigmascl = self.get_sigma_scaling(u[i,j,:],v[i,j,:],vis[i,j,:],sigma[i,j,:])
                    sigma[i,j,:] /= sigmascl
                    u[i,j,:] /= wav[i,j]
                    v[i,j,:] /= wav[i,j]
                else:  
                    u[i,j,:]    = evil.load_binary(direct+'u_spw_{0}_chan_{1}.bin'.format(i,j))*wav[i,j]
                    v[i,j,:]    = evil.load_binary(direct+'v_spw_{0}_chan_{1}.bin'.format(i,j))*wav[i,j]
                    vist  = evil.load_binary(direct+'vis_spw_{0}_chan_{1}.bin'.format(i,j))
                    vis[i,j,:] = vist[::2]+1j*vist[1::2]
                    sigma[i,j,:]= evil.load_binary(direct+'sigma_spw_{0}_chan_{1}.bin'.format(i,j))
                    ant1[i,j,:] = np.rint(evil.load_binary(direct+'ant_1.bin')).astype(int)
                    ant2[i,j,:] = np.rint(evil.load_binary(direct+'ant_2.bin')).astype(int)
                    chan[i,j,:] +=j+Nchan*i
                    sigmascl = self.get_sigma_scaling(u[i,j,:],v[i,j,:],vis[i,j,:],sigma[i,j,:])
                    sigma[i,j,:] /= sigmascl
                    u[i,j,:] /= wav[i,j]
                    v[i,j,:] /= wav[i,j]
                    time[i,j,:] = evil.load_binary(direct+'time.bin')
                    
        print "loaded all necessary files \n"
        if bintime is not None:  # function doesn't work for now, but this is where it'll happen.
            print "binning visibilities in intervals of {0} seconds".format(bintime)
            u,v,vis,sigma,ant1,ant2 = self.bin_data(u,v,wav,vis,sigma,ant1,ant2,time,bintime)
         
        print "building dOdphase \n"
        
        ant1 = np.rint(ant1.flatten()).astype('int')
        ant2 = np.rint(ant2.flatten()).astype('int')
        time = time.flatten()
        rowisone,colisone,rowisminusone,colisminusone = self.build_dOdphase(ant1,ant2,time,NUM_TIME_STEPS)
        print "dOdphase built, writing data to disk"
        sigma = sigma.flatten()
        u = u.flatten()
        v = v.flatten()
        vis = vis.flatten()
        chan=chan.flatten()
        
        Vis = np.zeros(2*len(vis),float)
        Vis[::2] = vis.real
        Vis[1::2]= vis.imag
        ssqinv = np.zeros(2*len(sigma),float)
        ssqinv[::2] = sigma**(-2)
        ssqinv[1::2]= sigma**(-2)
                
        return Vis , ssqinv , u , v , rowisone , colisone , rowisminusone , colisminusone , chan

# ---------------------------------------------------------------------------
    def bin_data(self,u,v,wavelength,vis,sigma,ant1,ant2,time,bintime):
        '''
        Bin the data based on the baseline.  Bintime is given seconds, and determines
        what is the maximum length of time that can be binned.  For any data whose
        uv distance varies too much over that bintime, the interval will be broken up
        into intervals across which it is ok to bin the visibilities.
        (this function exists because CASA does things wrong).
        '''
        raise Exception("Cannot bin data this way yet.  Set bintime=None in your function call. \n")
        omega_Earth = 7.27e-5
        Nspw = u.shape[0]
        Nchan= u.shape[1]
        Nant = int(max(ant2))
        
        for i in range(Nspw):
            for j in range(Nchan):
                
                # get list of baselines 
                baselines = [[i , j] for i in range(max(Nant)) for j in range(i+1,max(Nant))]
                for k in range(len(baselines)):
                    these = (ant1[i,j,:] == baselines[k,0]) & (ant2[i,j,:] == baselines[k,1])
                    uvdist= np.mean(u[i,j,these]**2+v[i,j,these]**2)
                    dist_lambda = uvdist /wavelength[i,j]
                    theta_FOV = 1.45e-4
                    tau_avg = 1.0/(omega_Earth * dist_lambda * theta_FOV)
                    
                    inds2 = np.where(np.diff(time[these])>bintime)
                    inds1 = 0
        
        return u , v , vis , sigma , ant1 , ant2
        
# ---------------------------------------------------------------------------

    def concatenate_spws(self,filedirslist,outputdir,samechan=True):
        '''
        Accepts list of directories to binary data (fully reduced).
        Joins multiple spectral windows together to make final dataset.
        '''
            
        NUM_SPW = len(filedirslist)
            
        for i in range(NUM_SPW):
            Dir = filedirslist[i]
            if i ==0:
                u = evil.load_binary(Dir+'u.bin')
                v = evil.load_binary(Dir+'v.bin')
                vis=evil.load_binary(Dir+'vis_chan_0.bin')
                sig=evil.load_binary(Dir+'sigma_squared_inv.bin')
                chan=evil.load_binary(Dir+'chan.bin')
                ro1 =evil.load_binary(Dir+'ROWisone.bin')
                co1 =evil.load_binary(Dir+'COLisone.bin')
                rom1=evil.load_binary(Dir+'ROWisminusone.bin')
                com1=evil.load_binary(Dir+'COLisminusone.bin')
            else:
                u = np.append(u,evil.load_binary(Dir+'u.bin'))
                v = np.append(v,evil.load_binary(Dir+'v.bin'))
                vis=np.append(vis,evil.load_binary(Dir+'vis_chan_0.bin'))
                sig=np.append(sig,evil.load_binary(Dir+'sigma_squared_inv.bin'))
                ro1 =np.append(ro1,evil.load_binary(Dir+'ROWisone.bin'))
                co1 =np.append(co1,evil.load_binary(Dir+'COLisone.bin'))
                rom1=np.append(rom1,evil.load_binary(Dir+'ROWisminusone.bin'))
                com1=np.append(com1,evil.load_binary(Dir+'COLisminusone.bin'))
                if samechan is True:
                    chan=np.append(chan,evil.load_binary(Dir+'chan.bin'))
                else:
                    chan=np.append(chan,evil.load_binary(Dir+'chan.bin')+i)
                    
        evil.write_binary(u,outputdir+'u.bin')
        evil.write_binary(v,outputdir+'v.bin')
        evil.write_binary(vis,outputdir+'vis_chan_0.bin')
        evil.write_binary(sig,outputdir+'sigma_squared_inv.bin')
        evil.write_binary(chan,outputdir+'chan.bin')
        evil.write_binary(ro1,outputdir+'ROWisone.bin')
        evil.write_binary(co1,outputdir+'COLisone.bin')
        evil.write_binary(rom1,outputdir+'ROWisminusone.bin')
        evil.write_binary(com1,outputdir+'COLisminusone.bin')
            
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
            print( "getting phases" )                                
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
        -v determines the rate at which the phase grid translates, in 
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
        data = struct.pack('d'*len(rowisone),*rowisone)
        f.write(data)
        f.close()
        
        g = open(str(datadir)+'COLisone.bin','wb')
        data = struct.pack('d'*len(colisone),*colisone)
        g.write(data)
        g.close()
        
        h = open(str(datadir)+'ROWisminusone.bin','wb')
        data = struct.pack('d'*len(rowisminusone),*rowisminusone)
        h.write(data)
        h.close()
        
        i = open(str(datadir)+'COLisminusone.bin','wb')
        data = struct.pack('d'*len(colisminusone),*colisminusone)
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