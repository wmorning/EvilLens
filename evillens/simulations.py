'''
Author:  Warren Morningstar

A simple-ish python script to convert CASA visibility data (contained in a
measurement set) to the binary format required by the Ripples analysis
pipeline.

All functions are defined at the beginning of the "main" script, and the 
code that is executed is at the end.  Arguments that should be edited
are at the beginning of the __main__ function.  Otherwise everything
should be left untouched or edited at the users own risk.
'''

import numpy as np
import os
import struct
from scipy.interpolate import RectBivariateSpline
import shutil


# ------------------------------------------------------------------------

def get_sigma_scaling(u,v,vis,sigma):
    '''
    Calculate the amplitude A that the system temperatures must be 
    scaled by in order to reflect the true noise in the data.
    
    Takes:
    
    u:     The u coordinates of the data (in meters)
    
    v:     The v coordinates of the data (in meters)
    
    vis:   The visibility data (in Jy), complex format
    
    sigma: The system temperatures data.
    
    Returns:
    
    A:     The noise scaling.
    
    '''
    
    UVMAX = np.max([abs(u),abs(v)]) * 1.01
    Bins  = np.arange(-UVMAX,UVMAX+12.,12.)
    
    # Count number of visibilities in each bin
    P,reject1,reject2 = np.histogram2d(u,v,bins=Bins)
    
    # Keep only bins that contain visibilities
    [row,col] = np.where(P!=0)
    
    # Keep track of stats (just in case something weird is happening)
    NumSkippedBins = 0
    TotalUsed      = 0
    
    # Setup real and imaginary noise arrays
    noise_r = np.zeros(u.shape)
    noise_i = np.zeros(v.shape)
    
    # Array for the indices of the visibilities that are subtracted
    indI = np.zeros(u.shape,int)
    
    
    # loop over bins
    for i in range(len(row)):
        
        # indices of visibilities in the bin
        inds = np.where((v>=Bins[col[i]]) & (v<Bins[col[i]+1]) & \
                        (u>=Bins[row[i]]) & (u<Bins[row[i]+1]))[0]
        

        # skip bins with only one visibility
        if len(inds) == 1:
            NumSkippedBins +=1
            continue
        
        elif len(inds) % 2 == 1: # discard extra visibility if needed 
            inds = inds[:-1]
        
        # get indices of visibilities to difference
        I = inds[::2]
        J = inds[1::2]
        

        # difference to get the noise
        noise_r[I] = (vis[I]-vis[J]).real
        noise_i[I] = (vis[I]-vis[J]).imag
        
        # record indices of subtracted visibilities
        indI[I] = J
        
        # track how much visibilities have been used (cumulatively)
        TotalUsed += len(inds)
        
    # keep only noise != 0 
    iI = np.where(np.logical_not( np.isclose(noise_r, 0)))[0]
    #iI = np.where(noise_r != 0)[0]
    N = len(iI)
    
    # Get A
    A = np.sqrt(N / np.sum((noise_r[iI]/np.sqrt(sigma[iI]**2+sigma[indI[iI]]**2))**2))
    
    # Verbose
    print "Total number of visibilities used was", TotalUsed , "out of" , len(vis)
    print NumSkippedBins , "bins out of" , len(row), "had only one visibility and were skipped"
    print "Sigma Scaling:  " , A
    print "sigma:  " , sigma
    
    return A
    
# ------------------------------------------------------------------------

def write_binary(array,outputfile,type = 'd'):
    '''
    A nice quick python function to write arrays as binary files.
    
    Takes:
    
    array:        A 1D array
    
    outputfile:  The file to write the data to.
    
    type:        'd' - double
                 'i' - int
    
    '''
    
    with open(outputfile,'wb') as file:
        if type == 'd':
            data = struct.pack('d'*len(array),*array)
            
        elif type == 'i':
            data = struct.pack('i'*len(array),*array)
            
        else:
            raise Exception('invalid type specified \n')
        
        file.write(data)
    
    file.close()
    
    return
      
# ------------------------------------------------------------------------

def Remove_missing_antennas(ant1,ant2):
    '''
    A function to remove antennas that are missing from the entire
    observation.  This is important to use in order to keep the dOdp 
    matrix from being potentially singular (because there is a phase
    parameter that is unconstrained by the data)
    
    Takes:
    
    ant1:     The first antenna used in each baseline
    
    ant2:     The second antenna used in each baseline
    
    returns:
    
    void:     No returned argument, but ant1 and ant2 will have been
              updated
    '''
    
    # Get a list of all antennas included in the dataset
    antennas = np.unique(np.append(np.unique(ant1),np.unique(ant2)))
    Nant = len(antennas)
    
    # Figure out which antennas are missing
    missing_antennas = np.setdiff1d(np.arange(np.max(antennas)),antennas)
    
    # Sort the missing antennas, so we can easily iterate over them
    missing_antennas = np.sort(missing_antennas)
    
    for i in range(len(missing_antennas)):
        # subtract 1 from the antenna IDs for all missing antennas 
        # with a lower ID
        ant1[ant1>missing_antennas[-(i+1)]] -= 1
        ant2[ant2>missing_antennas[-(i+1)]] -= 1
    
    print "these antennas are missing from your observation: " , missing_antennas
# ------------------------------------------------------------------------

def Build_dOdp(ant1,ant2,time,NUM_TIME_STEPS=1):
    '''
    Set up the dOdp files with a user input number of
    phase parameters.
    
    Takes:
    
    ant1:     The first antenna used in each baseline
    
    ant2:     The second antenna used in each baseline
    
    time:     The time stamp of each integration
    
    NUM_PHASE_PARS:   Number of phase intervals
    
    Returns:
    
    ID1:      The phase parameter IDs of the first antenna
    
    ID2:      The phase parameter IDs of the second antenna
    
    '''
    
    # Get rid of antennas not included in the ms
    Remove_missing_antennas(ant1,ant2)
    
    # reference numbers
    NUM_ANT = int(np.max([np.max(ant1),np.max(ant2)]))
    NUM_BL  = (NUM_ANT*(NUM_ANT-1))/2
    NUM_TIME_STEPS = int(NUM_TIME_STEPS)
    
    # to add:  Get rid of missing antennas in specific time interval 
    
    colisone      = np.zeros(len(ant1))
    rowisone      = np.zeros(len(ant1))
    colisminusone = np.zeros(len(ant2))
    rowisminusone = np.zeros(len(ant2))
    
    # Time cuts?
    tstart  = np.min(time)-1.0
    tend    = np.max(time)+1.0
    tdiff   = tend-tstart
    tswitch = tdiff/float(NUM_TIME_STEPS)
    
    
    
    for i in range(len(rowisone)):
        if ant1[i] == 0: # First antenna phase is always fixed to zero
            rowisminusone[i] = i
            colisminusone[i] = ant2[i]-1 + int((time[i]-tstart)//tswitch)*(NUM_ANT)
        else:
            rowisone[i]      = i
            colisone[i]      = ant1[i]-1 + int((time[i]-tstart)//tswitch)*(NUM_ANT)
            rowisminusone[i] = i
            colisminusone[i] = ant2[i]-1 + int((time[i]-tstart)//tswitch)*(NUM_ANT)
            
            
#   # ALTERNATIVE PHASE ASSIGNMENT.  USE SCAN FROM MS TO DETERMINE PHASE ASSOCIATION
#   scans = np.unique(scan)
#   for i in range(len(rowisone)):
#       if ant1[i] == 0:
#           rowisminusone[i] = i
#           colisminusone[i] = ant2[i]-1 + NUM_ANT * np.argmin(abs(scan[i]-scans))
#       else:
#           rowisone[i]      = i
#           colisone[i]      = ant1[i]-1 + NUM_ANT * np.argmin(abs(scan[i]-scans))
#           rowisminusone[i] = i
#           colisminusone[i] = ant2[i]-1 _ NUM_ANT * np.argmin(abs(scan[i]-scans))
            
    # Remove where ant1 is 0 from list
    colisone = colisone[(rowisone != 0)]
    rowisone = rowisone[(rowisone != 0)]
    
    return rowisone , colisone , rowisminusone , colisminusone
    
    
    return
    

# ------------------------------------------------------------------------

def get_phase_grid(antX,antY,time,amp,velocity,cellsize=10.0,randseed=1):
    """
    Given an array of antenna positions, observing time, wind velocity, and phase amplitude
    construct a simulated phase screen.
    """
    
    # useful iterables
    Nbaselines = len(antX)*(len(antX)-1)/2
    Ntsteps = len(time) // Nbaselines
    
    # duration of the observation, in seconds
    duration = np.max(time)-np.min(time)
    
    # determine the sizes of the phase screen
    maxX = (np.max(antX) +(np.max(antX)-np.min(antX))+velocity*duration) // cellsize * cellsize + 4*cellsize
    minX = np.min(antX) // cellsize * cellsize+ 4*cellsize
    maxY = 4 * (np.max(antY) // cellsize * cellsize + 4*cellsize)
    minY = 4 * (np.min(antY) // cellsize * cellsize + 4*cellsize)
    
    x = np.arange(minX,maxX+cellsize,cellsize)
    y = np.arange(minY,maxY+cellsize,cellsize)
    
    # get the initial_phase_screen (white noise)
    np.random.seed(randseed)
    phases = np.random.normal(0.0,1.0,(len(y),len(x)))
    
    # FFT the phase screen and get coordinates
    phases = np.fft.fft2(phases / cellsize)
    FreqX = np.fft.fftfreq(len(x),1.0/float(len(x)))*2.0*np.pi/(x[-1]-x[0])
    FreqY = np.fft.fftfreq(len(y),1.0/float(len(y)))*2.0*np.pi/(y[-1]-y[0])
    
    print phases.shape
    
    # Slowly transform the FT phase screen by the power spectrum
    for i in range(len(FreqX)):
        for j in range(len(FreqY)):
            if np.sqrt(FreqX[i]**2+FreqY[j]**2)>1.0/1000.0:
                phases[j,i] *=(np.pi/180.)*amp*np.sqrt(0.0365)*(1000.0*np.sqrt(FreqX[i]**2+FreqY[j]**2))**(-11.0/6.0)
            elif np.sqrt(FreqX[i]**2+FreqY[j]**2)<1.0/1000.0 and np.sqrt(FreqX[i]**2+FreqY[j]**2)>1.0/6000.:
                phases[j,i] *= (np.pi/180.)*amp*np.sqrt(0.0365)*(1000.0*np.sqrt(FreqX[i]**2+FreqY[j]**2))**(-5.0/6.0)
            else:
                phases[j,i] *= (np.pi / 180.)*np.sqrt(0.0365)*(amp)*(6.0)**(-5.0/6.0)
    
    phases = 4*np.pi*np.fft.ifft2(phases)
    return phases,x,y
    
# ------------------------------------------------------------------------

def assign_phases_to_antennas(ant1,ant2,antX,antY,PhaseGrid,phase_x,phase_y,velocity,time):
    '''
    Given antenna IDs, coordinates, and the phase grid and its coordinates:  Translate
    the antennas across the grid and record the phase for each antenna.  Returns a 1D array
    of antenna phases (uncalibrated) for the 1st and 2nd antenna in each observation
    '''
    
    f_interp = RectBivariateSpline(phase_y,phase_x,PhaseGrid,kx=1,ky=1)
    antenna1_phase = f_interp.ev(antY[ant1],antX[ant1]+velocity*time)
    antenna2_phase = f_interp.ev(antY[ant2],antX[ant2]+velocity*time)
    
    # Also want to get a 2d array of the phase of each antenna with time, to ease with calibration
    Ntsteps = len(time)/(len(antX)*(len(antX)-1)/2)
    antennaphases = np.zeros([len(antX),Ntsteps],float)
    tstepsize = np.unique(time)[1]-np.unique(time[0])
    for i in range(len(antX)):
        print antennaphases[i,:].shape, 
        antennaphases[i,:] = f_interp.ev(antY[i]*np.ones(Ntsteps),antX[i]+tstepsize*velocity*np.arange(Ntsteps))
    
    return antenna1_phase,antenna2_phase,antennaphases
    
# ------------------------------------------------------------------------    
    
def Mock_phase_calibration(antennaphases,ant1,ant2,pwv_mean,proportional_error):
    pwv = antennaphases / (2*np.pi) + pwv_mean
    pwv_meas = pwv + 1.0e-5 * np.random.normal(0.0,abs(1+pwv/0.001))
    WVR_correction = 2*np.pi*(pwv_meas-pwv_mean)
    
    PE = np.random.normal(0.0,proportional_error,antennaphases.shape[0])
    PE = np.outer(PE,np.ones(antennaphases.shape[1]))
    PE *= antennaphases
    
    # Total WVR estimated phase
    WVR_correction -= PE
    
    correction_ant1 = np.zeros(ant1.shape)
    correction_ant2 = np.zeros(ant2.shape)
    
    Nantennas = len(np.unique(np.append(np.unique(ant1),np.unique(ant2))))
    for i in range(Nantennas-1):
        for j in range(i+1,Nantennas):
            correction_ant1[np.logical_and((ant1==i),(ant2==j))] -= WVR_correction[i]
            correction_ant2[np.logical_and((ant1==i),(ant2==j))] -= WVR_correction[j]
    return correction_ant1 , correction_ant2
