'''
Author:  Warren Morningstar

A script to be run from within CASA (I used 4.2.2, just FYI) that 
runs minimal input gravitational lens simulations and generates 
ALMA visibilities (including time variable phase errors, and their calibration)

Unfortunately, I had to define a number of functions in the middle of the code, which 
I was hoping to avoid to make the code appear less daunting.  The good news is that
a user should never have to interface with these functions at all.  They are 
intended to be complete as is.  The same is true with the __main__ function, 
which performs all of the simulations, but contains none of the control parameters.

Instead, users are advised to only edit the first 4 blocks of arguments 
(lens simulation Arguments, CASA simulation arguments, Phase simulation arguments,
& Visibility Binning Arguments).  The arguments contained there should be able to
control all necessary aspects of the simulation.

The outputs of this pipeline are a folder containing binary files of all the 
necessary inputs to the Ripples pipeline, as well as a subfolder containing 
the blinded parameters.  The subfolder should not be examined unless a user
is ready to deblind.
'''

import numpy as np
import sys
sys.path.append('/Users/wmorning/research/EvilLens') # Set to your location of EvilLens
import evillens as evil 
from evillens.simulations import *
import shutil
import os
import matplotlib.pyplot as plt

# reset mstransform,simobserve to defaults just in case
default('mstransform')
default('simobserve')


# ----------------- lens simulation Arguments ----------------- #

# parameter ranges
gamma_range = [0.4,1.8]                     #
mass_range  = [1.12,1.13]                   #
elp_range   = [0.0,0.25]                    #
ang_range   = [-np.pi,np.pi]                #
x_range     = [-0.1,0.1]                    # 
y_range     = [-0.1,0.1]                    #
shear_range = [-0.05,0.05]                  #
mult_range  = [-0.05,0.05]                  #
zlens       = 0.14                          # Trying to decide if these should be inputs or randoms
zsrc        = 4.0                           # Same as the lens redshift.

# random seed for control
lens_seed = 48214

# how complicated is lens model?
SIE = True                                  # if false, a gamma will be randomly chosen, otherwise gamma = 1
Multipoles = False                          # if false, just add shear.  Otherwise add m = 3 and m = 4 multipoles.
Substructures = False                       # maybe this shouldn't be an input, and the code should randomly decide.

# lens grid parameters
NX_lens = 400                               # Note that the full extent of the lensed image grid is
NY_lens = 400                               # NX * pixscale_lens, so make sure that this is at least 4 arcsec
pixscale_lens = 0.01                        # Also, remember that ALMA 6k or 12k bl can resolve 0.002 arcsec...

# Source parameters
source_scale = 1.0                          # In case we want to stretch the size of the source grid
source_grid_center_range = [-0.4,0.4]       # Randomly pick a position for the source
build_source = True                         # If true, generates the source image (slow)
source_image = '../CarterPewterSchmidt_src.fits' # If build_source is false, you must load one (fits image)


# ----------------- CASA simulation Arguments ----------------- #
incenter = '345GHz'                         # Observing frequency
inwidth = '8GHz'                            # Observing bandwidth (ALMA continuum is 7.5 GHz)
Flux = 0.15                                 # Flux (in Jy)
indirection = 'J2000 19h00m00 -50d00m00'    # Location of source on sky (not sure if actually used)
project = 'Practice_simulation'             # Name of simulation
antennalist = 'alma.cycle2.7.cfg'           # ALMA observing configuration
totaltime = '40min'                         # Duration of observation
integration = '60s'                         # Time for single ALMA integration
skymodel = 'Practice_simulation.fits'       # Name for lensed image fits file (writes and reads this image)

# ----------------- Phase Error Sim Arguments ----------------- #

add_phase_errors = False                    # if false, can ignore all other args here
Mock_calibration = True                     # if True, a Mock ALMA WVR calibration is used
Phase_random_seed = 1                       # random seed for control
wind_speed=6.0                              # velocity of the phase screen (in m/s)
Phase_amp=100.0                             # amplitude of the phase screen (in degrees)
pwvmean = 0.003                             # mean pwv column height (in meters... used in phase calibration)
proportional_error = 0.01                   # proportional error of WVR calibration

# --------------- Visibility Binning Arguments ---------------- #

run_mstransform   = True                    # if False, does not run mstransform
datacolumn     = 'all'                      # set to all
chanaverage    = False                      # If multiple channels, we may want to average
chanbin        = 480                        # number of input channels to form output channel
timeaverage    = True                       # If long duration observation, we may want to time average
timebin        = '600s'                     # Size of time bins
timespan       = 'state,scan'               # Possibly meaningless
maxuvwdistance = 6.0                        # max distance antennas are allowed to move in uv plane (keep <= 12)
spw            = ''                         # Which spectral windows to include (use '' for all)
keepflags      = False                      # Keep flagged data? (obviously keep False)
    
    
# post mstransform arguments
NUM_TIME_STEPS = 1                          # Number of time intervals to use in Ripples phase calibration

# ------------------------------------------------------------- #

# ========================================================================

# Some functions have to be defined here because they interact with CASA's ms, tb, and msmd tools
# Blech, but we'll have to live with it unless we want to pass those tools as arguments

def add_phase_errors_and_noise(noiseless_vis,noisy_vis,outputvis,amp,velocity,Mock_cal=False,pwv_mean=0.003,proportional_error=0.02,randseed=1):
    '''
    Take a noiseless measurement set, and sabotage it with phase errors and noise.
    This should be done for each channel individually, although the phases should be scaled
    by wavelength (which is conveniently contained within the ms).
    '''
    # link to metadata (gives us spw and channel information)
    msmd.open(noiseless_vis)
    
    # get number of spw
    NSPW      = msmd.nspw()
    
    shutil.copytree(noiseless_vis,outputvis)
    
    for spw in range(NSPW):  # Iterate over spectral windows.
        
        NCHAN = msmd.nchan(spw)
        chan_freqs = msmd.chanfreqs(spw)
        
        for chan in range(NCHAN):  # Iterate over channels.
            # Load all this spw/channel's data 
            ms.open(noiseless_vis)
            ms.msselect({'spw':str(spw)})
            u = ms.getdata("UVW")["uvw"][0]
            v = ms.getdata("UVW")["uvw"][1]
            vis = ms.getdata("data")["data"][:,chan,:]
            ant1 = ms.getdata("antenna1")["antenna1"]
            ant2 = ms.getdata("antenna2")["antenna2"]
            time = ms.getdata("time")["time"]
            time -= np.min(time)
            frequency = msmd.chanfreqs(spw)[chan]
            wavelength = 3.0*10**8 / frequency
            ms.close()
            
            # Get info about the antenna positions for phase errors (I love the ALMA stores this info!!!)
            tb.open(noiseless_vis+"/ANTENNA")
            antX = np.squeeze(tb.getcol('POSITION')[0])
            antY = np.squeeze(tb.getcol('POSITION')[1])
            tb.close()
            antX -= np.mean(antX)
            antY -= np.mean(antY)
            
            if (spw ==0) & (chan ==0):
                PhaseGrid,phase_x,phase_y = get_phase_grid(antX,antY,time,amp,velocity,10.0,randseed)
                antenna1_phase , antenna2_phase , antennaphases = assign_phases_to_antennas(ant1,ant2,antX,antY,PhaseGrid,phase_x,phase_y,velocity,time)
                
                # and now get the phase calibration, if requested
                if Mock_cal ==True:
                    Pcor1 , Pcor2 = Mock_phase_calibration(antennaphases,ant1,ant2,pwv_mean,proportional_error)
                
            # Before applying the phase errors, lets subtract the noiseless visibilities from the noisy ones to get the noise
            ms.open(noisy_vis)
            ms.msselect({'spw':str(spw)})
            sigma = ms.getdata("data")["data"][:,chan,:]-vis
            ms.close()
            
            # Have phase (unscaled by wavelength), now we apply the phase error to the visibilities, scaling by wavelength
            vis[0,:] *= np.exp(1j*(antenna1_phase-antenna2_phase)/wavelength)
            vis[1,:] *= np.exp(1j*(antenna1_phase-antenna2_phase)/wavelength)
            
            # Also add the noise
            vis += sigma
            
            # Apply the phase calibration
            if Mock_cal ==True:
                vis[0,:] *= np.exp(1j*(Pcor1-Pcor2)/wavelength)
                vis[1,:] *= np.exp(1j*(Pcor1-Pcor2)/wavelength)
            
            # have corrupted visibilities.  Now write them to the phase_err ms
            ms.open(outputvis,nomodify=False)
            ms.msselect({'spw':str(spw)})
            datatoput = ms.getdata("data")
            datatoput["data"][:,chan,:] = vis
            ms.putdata(datatoput)
            ms.close()
            
            
def ms_to_bin(MeasurementSet,outputdir):
    '''
    This code takes a measurement set, goes through its contents, and spits the 
    result out into the correct Ripples files in the location specified by filename_prefix
    '''
    
    # link to metadata (gives us spw and channel information)
    msmd.open(MeasurementSet)
    
    # get number of spw
    NSPW      = msmd.nspw()
    
    # have to count channels in a weird way, just in case
    channel = 0
    
    for i in range(NSPW):  # Iterate over spectral windows.
        
        NCHAN = msmd.nchan(i)
        chan_freqs = msmd.chanfreqs(i)
        
        for j in range(NCHAN):  # Iterate over channels.
        
            ms.open(MeasurementSet)
            ms.msselect({'spw':str(i)})
            
            # get desired data
            freqij = chan_freqs[j]
            uij    = ms.getdata("UVW")["uvw"][0] * freqij / (3.*10**8)
            vij    = ms.getdata("UVW")["uvw"][1] * freqij / (3.*10**8)
            visij  = ms.getdata("data")["data"][:,j,:]
            sigmaij= ms.getdata("sigma")["sigma"]
            ant1ij = ms.getdata("antenna1")["antenna1"]
            ant2ij = ms.getdata("antenna2")["antenna2"]
            timeij = ms.getdata("time")["time"]
            
            ms.close()
            
            # average the polarizations (weighted by noise to tease out the signal a bit better)
            visij   = np.average(visij,weights = sigmaij**-2.,axis=0) 
            sigmaij = np.sum(sigmaij**-2.,axis=0)**-0.5
            
            
            
            # scale the noise
            sigmaij /= get_sigma_scaling(uij/(freqij / (3.*10**8)),vij/(freqij / (3.*10**8)),visij,sigmaij) 
            
            # Build dOdp if need be (might not even use it)
            # Build_dOdp(ant1,ant2,time,NUM_PHASE_PARS)  # function not written yet
            
            # build up the full data files
            if (i==0) & (j==0):
                vis   = np.copy(  visij  )
                u     = np.copy(   uij   )
                v     = np.copy(   vij   )
                sigma = np.copy( sigmaij )
                ant1  = np.copy(  ant1ij )
                ant2  = np.copy(  ant2ij )
                time  = np.copy(  timeij )
                chan  = np.zeros(len(uij))
                
            else:
                vis   = np.append(vis   ,   visij)
                u     = np.append(u     ,     uij)
                v     = np.append(v     ,     vij)
                sigma = np.append(sigma , sigmaij)
                ant1  = np.append(ant1  ,  ant1ij)
                ant2  = np.append(ant2  ,  ant2ij)
                time  = np.append(time  ,  timeij)
                chan  = np.append(chan  ,  channel *np.ones(len(uij)))
               
            channel += 1
            
    # if outputdir doesn't exist, make it
    # Should be mkdirs but CASA doesnt have that....
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        
    # Get vis and sigma into all real (Ripples) format
    Vis = np.zeros(2*len(vis))
    Sigma=np.zeros(2*len(sigma))
    Vis[::2]    = vis.real
    Vis[1::2]   = vis.imag
    Sigma[::2]  = sigma**-2.
    Sigma[1::2] = sigma**-2.
    
    # Build the DOdPhase matrix (this is for older versions of Ripples)
    ROWisone , COLisone , ROWisminusone , COLisminusone = Build_dOdp(ant1,ant2,time,NUM_TIME_STEPS)
    
    # write everything to binary files
    write_binary(Vis , outputdir+"vis_chan_0.bin")
    write_binary(u ,            outputdir+"u.bin")
    write_binary(v ,            outputdir+"v.bin")
    write_binary(Sigma,outputdir+"sigma_squared_inv.bin")
    write_binary(ant1,       outputdir+"ant1.bin")
    write_binary(ant2,       outputdir+"ant2.bin")
    write_binary(time,       outputdir+"time.bin")
    write_binary(chan,       outputdir+"chan.bin")
    write_binary(ROWisone,   outputdir+"ROWisone.bin")
    write_binary(COLisone,   outputdir+"COLisone.bin")
    write_binary(ROWisminusone, outputdir+"ROWisminusone.bin")
    write_binary(COLisminusone, outputdir+"COLisminusone.bin")


# ========================================================================

if __name__ == '__main__':
    '''
    A large and involved pipeline for simulating visibilities.  The steps involved in the pipeline 
    are:
    
    1)      Create mock image of lens (possibly containing substructures)
    2)      Writing mock image to fits file
    3)      Run the CASA task simobserve to get the ALMA measurement set
    4)      Read in the measurement set (in channel blocks just in case), and add phase errors and noise
    5)      Bin the sabotaged measurement set spectrally and temporally
    6)      Estimate the noise level empirically
    7)      Write the binned dataset to the binary files that will be read in by the pipeline
    8)      Write the true parameter information to a locked directory for future deblinding
    8)      Clean up by removing all measurement sets and mezanine files.
    '''
    
    print("Starting the lensing simulation")
    
    # seed the random variable
    np.random.seed(lens_seed)
    
    # Generate the lens parameters (even those we won't use)
    Gamma= np.random.random()*(gamma_range[1]-gamma_range[0])+gamma_range[0]
    logM = np.random.random()*(mass_range[1]-mass_range[0]) + mass_range[0]
    elp  = np.random.random()*(elp_range[1] -elp_range[0] ) + elp_range[0]
    angle= np.random.random()*(ang_range[1] -ang_range[0] ) + ang_range[0]
    xl   = np.random.random()*( x_range[1]  -  x_range[0] ) + x_range[0]
    yl   = np.random.random()*( y_range[1]  -  y_range[0] ) + y_range[0]
    g1   = np.random.random()*(shear_range[0]-shear_range[1])+shear_range[0]
    g2   = np.random.random()*(shear_range[0]-shear_range[1])+shear_range[0]
    A3   = np.random.random()*(mult_range[0]-mult_range[1]) + mult_range[0]
    B3   = np.random.random()*(mult_range[0]-mult_range[1]) + mult_range[0]
    A4   = np.random.random()*(mult_range[0]-mult_range[1]) + mult_range[0]
    B4   = np.random.random()*(mult_range[0]-mult_range[1]) + mult_range[0]
    
    # create the lens
    lens = evil.PowerKappa(zlens,zsrc)
    lens.setup_grid(NX=NX_lens,NY=NY_lens,pixscale=pixscale_lens)
    lens.build_kappa_map(logM=logM,q=1-elp,angle=angle,centroid=[xl,yl],Gamma=Gamma)
    lens.deflect()
    lens.add_multipoles([[g1,g2],[Multipoles*A3,Multipoles*B3],[Multipoles*A4,Multipoles*B4]])
    
    # Construct the source
    lens.source = evil.Source(zsrc)
    if build_source is True:
        lens.source.setup_grid(NX=1600,NY=1600,pixscale=0.000625)  # need crazy hi-res grid because of magnification
        
        # I have found that these control parameters for the source seem to produce good images
        Nclumps = np.random.randint(1,5)
        Nsubclumps = np.random.randint(100,500)
        xs = 0.0
        ys = 0.0
        qs = np.random.random()*0.7+0.3
        phi_s = np.random.random()*np.pi
        r_hl_s = np.random.random()*0.03+0.02
        n_src = np.random.random()*0.75+0.25
        
        # Create the source image
        lens.source.build_sersic_clumps(Nnuclei=Nclumps,NclumpsPerNucleus=Nsubclumps,\
                                        x0=xs,y0=ys,q=qs,phi=phi_s,r_hl=r_hl_s,n=n_src,\
                                        seed1=lens_seed+500)
    else:
        lens.source.read_source_from(source_image)
    
    # scale the source grid
    lens.source.beta_x *= source_scale
    lens.source.beta_y *= source_scale
    
    # shift center position of source grid (and thus, the position of the source)
    source_grid_center = np.random.random(2)*(source_grid_center_range[1]-source_grid_center_range[0])+source_grid_center_range[0]
    lens.source.beta_x += source_grid_center[0]
    lens.source.beta_y += source_grid_center[1]
    
    # Add subhalos if requested ---- For now lets just leave it blank (we need to decide how to add subhalos)
    #if Substructures is True:
    #    lens.add_subhalos()
    
    # Perform the raytracing to get the lensed image
    lens.raytrace()
    
    # For now, just to debug, lets see if it gets this far!!
    #lens.plot('non-lensed image')


    # Write the image to fits file to be simobserve'd
    lens.write_image_to(skymodel)
    
    print("lens successfully written to data, creating the visibilities")
    
    # get image flux for simobserve
    inbright = str(Flux*np.max(lens.image)/np.sum(lens.image))+'Jy/pixel'
    incell   = str(lens.pixscale)+'arcsec'

    # Create the visibilities
    simobserve(incenter=incenter,inwidth=inwidth,inbright=inbright,indirection=indirection,antennalist=antennalist,\
                totaltime=totaltime,integration=integration,incell=incell,project=project,skymodel=skymodel)
                
    print("Visibilities have been created, adding Phase errors (if requested)")
    
    # Simobserve creates deterministic filenames.  So lets exploit that to define relevant files here
    noisyms = project+'/'+project+'.'+antennalist[:-3]+'noisy.ms'
    noiselessms = project+'/'+project+'.'+antennalist[:-3]+'ms'

    # Create a new measurement set path for the phase calibrated ms
    Phase_error_ms   = project+'/'+project+'.'+antennalist[:-3]+'sabotaged.ms'

    # add the phase errors if requested
    if add_phase_errors is True:
        add_phase_errors_and_noise(noiselessms,noisyms,Phase_error_ms,Phase_amp,wind_speed,Mock_calibration,pwvmean,proportional_error,Phase_random_seed)
        current_ms = Phase_error_ms
    else:
        current_ms = noisyms

    # Binned measurement set name
    Binned_ms = project+'/'+project+'.'+antennalist[:-3]+'binned.ms'

    # output file prefix for binary files
    output_file_prefix = project+'/'+project+'/'

    if run_mstransform:
        mstransform(vis=current_ms,outputvis=Binned_ms,datacolumn=datacolumn,chanaverage=chanaverage,chanbin=chanbin,timeaverage=timeaverage,timebin=timebin,maxuvwdistance=maxuvwdistance,spw=spw)

        current_ms = Binned_ms

    # write the measurement set to binary files
    ms_to_bin(current_ms,output_file_prefix)
    
    # Write blinded parameters to a file
    #write_blinded_parameters(lens,output_file_prefix+'Top_secret/')
    
    print("Data has been written to the output folder, now its time to clean up")
    
    # Clean up (remove temporary files)
    shutil.rmtree(noisyms)
    shutil.rmtree(noiselessms)
    if add_phase_errors is True:
        shutil.rmtree(Phase_error_ms)
    if run_mstransform is True:
        shutil.rmtree(Binned_ms)
    
    
    print("The simulation pipeline has run successfully!!")