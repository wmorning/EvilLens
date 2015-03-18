# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 15:06:23 2015

@author: wmorning
"""
# ===========================================================================

import numpy as np
import evillens as evil

# ===========================================================================

class Saboteur(object):
    '''
    An object class which sabotages Visibilities data
    '''

    def __init__(self):
        
        self.Visibilities = None
        self.antenna1 = None
        self.antenna2 = None
        self.u = None
        self.v = None
        
        self.W = 1000.0
        self.L0 = 6000.0
        return
    
# ---------------------------------------------------------------------------
    
    def read_data_from(self, MeasurementSet):
        raise Exception("cannot read in data yet.\n")
        
# ---------------------------------------------------------------------------        

    def add_decoherence(self, K , wavelength):
        '''
        Add decoherence to visibilities.  Use baseline to determine rms phase 
        error and convolve the visibilities using equation 4 in Carilli and 
        Holdaway.
        '''
        b = np.sqrt(self.u**2+self.v**2)
    
        for i in range(len(b)):
            if b[i] <= self.W:
                self.Visibilities[i] *= np.exp(-((K/wavelength)*(b[i]/1000)**(5.0/6.0)*(np.pi/180.0))**2/2)
            elif b[i] >self.W and b[i] <= self.L0:
                self.Visibilities[i] *= np.exp(-((K/wavelength)*(b[i]/1000)**(1.0/3.0)*(np.pi/180.0))**2/2)
            else:
                self.Visibilities[i] *= np.exp(-((K/wavelength)*(np.pi/180.0))**2/2)
            
        return

# ---------------------------------------------------------------------------
    
def add_amplitude_errors(self, rms_error):
    '''
    Add amplitude errors to the visibilities.  Each antenna gets gaussian
    random amplitude error centered around 1 with rms equal to input rms. 
    '''
    errors = np.random.normal(1.0, rms_error, int(max(self.antenna2)+1.01))
    
    visibilities_new = np.zeros(len(self.Visibilities),complex)
    
    for i in range(len(visibilities_new)):
        visibilities_new[i] = self.Visibilities[i]*errors[self.antenna1[i]]*errors[self.antenna2[i]]
        
    self.Visibilities=visibilities_new

# ---------------------------------------------------------------------------    
def add_phase_errors(self):
    '''
    Create coordinate grid of rms phases, using the phase structur function.
    Assign one phase to each antenna, determined using the antenna's position
    on the grid.  All visibilities are shifted by the phases assigned to each
    of their antennas.
    '''
    
    raise Exception("cannot add phase errors yet.\n")
    
        
    
    