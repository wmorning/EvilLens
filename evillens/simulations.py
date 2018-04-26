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
from xml.etree.ElementTree import Element , SubElement , Comment
from xml.etree import ElementTree
from xml.dom import minidom
import json


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
    maxY = np.max([6000.//cellsize * cellsize+cellsize,4 * (np.max(antY) // cellsize * cellsize + 4*cellsize)])
    minY = np.min([-6000.//cellsize * cellsize+cellsize,4 * (np.min(antY) // cellsize * cellsize + 4*cellsize)])
    
    x = np.arange(minX,maxX+cellsize,cellsize)
    y = np.arange(minY,maxY+cellsize,cellsize)
    
    # get the initial_phase_screen (white noise)
    np.random.seed(randseed)
    phases = np.random.normal(0.0,1.0,(len(y),len(x)))
    
    # FFT the phase screen and get coordinates
    phases = np.fft.fft2(phases / cellsize)
    FreqX = np.fft.fftfreq(len(x),1.0/float(len(x)))*2.0*np.pi/(x[-1]-x[0])
    FreqY = np.fft.fftfreq(len(y),1.0/float(len(y)))*2.0*np.pi/(y[-1]-y[0])
    
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
        antennaphases[i,:] = f_interp.ev(antY[i]*np.ones(Ntsteps),antX[i]+tstepsize*velocity*np.arange(Ntsteps))
    
    return antenna1_phase,antenna2_phase,antennaphases
    
# ------------------------------------------------------------------------    
    
def Mock_phase_calibration(antennaphases,ant1,ant2,pwv_mean,proportional_error):
    
    # Change from electrical path length to meters and add the mean pwv
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
    
    
def Write_xml_file():
    '''
    Given a lens object, as well as a number of other arguments, create the tree structure for
    the xml file  that is read by the pipeline.
    '''
    
    # Create the tree structure
    Ripples               = Element('Ripples')
    parameters            = SubElement(Ripples,'parameters')
    num_gangs             = SubElement(parameters,'num_gangs')
    num_chan              = SubElement(parameters,'num_chan')
    zLENS                 = SubElement(parameters,'zLENS')
    zSOURCE               = SubElement(parameters,'zSOURCE')
    imside                = SubElement(parameters,'imside')
    LAMBDA                = SubElement(parameters,'LAMBDA')
    file_name             = SubElement(parameters,'file_name')
    mask_file             = SubElement(parameters,'mask_file')
    num_side              = SubElement(parameters,'num_side')
    N_sg_pix              = SubElement(parameters,'N_sg_pix')
    SRC_L                 = SubElement(parameters,'SRC_L')
    lens_centerX          = SubElement(parameters,'lens_centerX')
    lens_centerY          = SubElement(parameters,'lens_centerY')
    primary_beam_fwhm     = SubElement(parameters,'primary_beam_fwhm')
    src_centX             = SubElement(parameters,'src_centX')
    src_centY             = SubElement(parameters,'src_centY')
    src_prior_type        = SubElement(parameters,'src_prior_type')
    subhalo               = SubElement(Ripples,'subhalo')
    phase_angle_prior_rms = SubElement(subhalo,'phase_angle_prior_rms')
    num_phase_pars        = SubElement(subhalo,'num_phase_pars')
    logMsub               = SubElement(subhalo,'logMsub')
    dOdPhase_file         = SubElement(subhalo,'dOdPhase_file')
    subhaloX_file         = SubElement(subhalo,'subhaloX_file')
    subhaloY_file         = SubElement(subhalo,'subhaloY_file')
    output_file           = SubElement(subhalo,'output_file')
    subhalo_start_ind     = SubElement(subhalo,'subhalo_start_ind')
    deltaP                = SubElement(subhalo,'deltaP')
    action                = SubElement(Ripples,'action')
    Model                 = SubElement(action,'Model')
    modelpars             = SubElement(action,'modelpars')
    parameter_flags       = SubElement(action,'parameter_flags')
    stepsize              = SubElement(action,'stepsize')
    task                  = SubElement(action,'task')
    findLambda            = SubElement(action,'findLambda')
    output_file_prefix    = SubElement(action,'output_file_prefix')
    GetImage              = SubElement(action,'GetImage')
    PhaseCal              = SubElement(action,'PhaseCal')
    MCMC                  = SubElement(Ripples,'MCMC')
    numWalkers            = SubElement(MCMC,'numWalkers')
    prior_up              = SubElement(MCMC,'prior_up')
    prior_dn              = SubElement(MCMC,'prior_dn')
    start_rms             = SubElement(MCMC,'start_rms')
    TEMPERATURE           = SubElement(MCMC,'TEMPERATURE')
    output_filename       = SubElement(MCMC,'output_filename')
    resume_filename       = SubElement(MCMC,'resume_filename')
    NumIter               = SubElement(MCMC,'NumIter')
    FileWriteNumIter      = SubElement(MCMC,'FileWriteNumIter')
    PhaseCal2             = SubElement(MCMC,'PhaseCal')
    
    # populate the tree with the relevant arguments
    num_gangs.text             = '1'
    num_chan.text              = '1'
    zLENS.text                 = '%.3f'%(lens.zd)
    zSOURCE.text               = '%.3f'%(lens.zs)
    imside.text                = 5.0
    LAMBDA.text                = 1.0e-17
    file_name.text             = 'data/visibility_data/'+output_file_prefix
    mask_file.text             = 'none'
    num_side.text              = '80'
    N_sg_pix.text              = '60'
    SRC_L.test                 = '%.1f'%(lens.pixscale * lens.NX)
    lens.centerX.text          = '%.3f'%(0.1*np.round(lens.centroid[0]/0.1))
    lens.centerY.text          = '%.3f'%(0.1*np.round(lens.centroid[0]/0.1))
    primary_beam_fwhm.text     = '%.3f'%(1.02 * wavelength / 12.0 * 3600. * 180. / np.pi)
    src_centX.text             = '%.3f'%()
    src_centY.text             = '%.3f'%()
    src_prior_type.text        = 'grad'
    phase_angle_prior_rms.text = '15.0'
    num_phase_pars.text        = str(NUM_TIME_STEPS)
    logMsub.text               = '9.0'
    dOdPhase_file.text         = 'data/visibility_data/'+output_file_prefix
    subhaloX_file.text         = 'data/visibility_data/'+output_file_prefix+'sub_x.bin'
    subhaloY_file.text         = 'data/visibility_data/'+output_file_prefix+'sub_y.bin'
    output_file.text           = 'data'
    subhalo_start_ind.text     = '0'
    deltaP.text                = '1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6'
    Model.text                 = 'PowerKappa'
    modelpars.text             = str(np.ones(12))
    parameter_flags.text       = '1 1 1 1 1 1 1 1 1 1 1 1'
    stepsize.text              = '0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1'
    task.text                  = 'EVALMOD'
    findLambda.text            = '1'
    output_file_prefix.text    = 'data/models/'+output_file_prefix[:-1]
    GetImage.text              = '1'
    PhaseCal.text              = '0'
    numWalkers.text            = '64'
    prior_up.text              = '2.0 2.0 10 10 10 10 1 1 1 1 1 1'
    prior_dn.text              = '0.0 1.0 -10 -10 -10 -10 -1 -1 -1 -1 -1 -1'
    start_rms.text             = '0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001'
    TEMPERATURE.text           = '1.0'
    output_filename.text       = 'data/mcmc_chains/'+output_file_prefix
    resume_filename.text       = 'none'
    NumIter.text               = '10000'
    FileWriteNumIter.text      = '10'
    PhaseCal2.text             = '0'
    
    
    
    # Create and write the xml file
    rough_string = ElementTree.tostring(Ripples,'utf-8')
    reparsed = minidom.parseString(rough_string)
    with open(filename,'w') as file:
        file.write(reparsed.toprettyxml(indent="    "))
    
    
def write_blinded_parameters(lens,output_file_prefix):
    '''
    Given a lens, write the lens information to a folder 
    '''
    
    if not os.path.exists(output_file_prefix):
        os.mkdir(output_file_prefix)
    
    lens.source.write_source_to(output_file_prefix+'source_true.fits')
    
    # Make a Dictionary of lens model parameters
    filecontents = {'gamma':lens.Gamma ,\
                    'logM':lens.logM ,\
                    'ex': (1-lens.q)*np.cos(lens.angle)*10,
                    'ey':-(1-lens.q)*np.sin(lens.angle)*10,
                    'x':lens.centroid[0],
                    'y':lens.centroid[1],
                    'g1':lens.Multipoles[0,0],
                    'g2':lens.Multipoles[0,1],
                    'A3':lens.Multipoles[1,0],
                    'B3':lens.Multipoles[1,1],
                    'A4':lens.Multipoles[2,0],
                    'B4':lens.Multipoles[2,1]}
    
    # Add the subhalo parameters to the dict
    
    # Add the phase hyperparameters to the list?
    
    #
    with open(output_file_prefix+'blinded_parameters.txt','w') as file:
        file.write(json.dumps(filecontents))
    
    
