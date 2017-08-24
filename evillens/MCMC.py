# -*- coding: utf-8 -*-
"""
Created on Tue February 2nd 15:14:00 2016

@author: warrenmorningstar
"""

import numpy as np
import matplotlib.pyplot as plt
import glob



class MCMC(object):
    '''
    An object class that can manipulate Blueberry MCMC outputs.
    '''
    
    def __init__(self):
        self.data = None
        self.Chi2 = None
        self.GR = None
        self.Nwalkers = None
        self.Nparameters=None
        self.Niter=None
        self.Postmean = None
        return
    
    def load_chains_from(self, targetdir):
        '''
        Load chains from target directory.  Assumes chains are written in the 
        Blueberry output format.
        '''
        filenames = glob.glob(targetdir+'chain_number_*')
        
        self.Nwalkers = len(filenames)
        
        for i,f in enumerate(filenames):
            temp = np.loadtxt(f)
            if i ==0:
                self.Niter = temp.shape[0]
                self.Nparameters = temp.shape[1]-1
                self.data = np.zeros([self.Nwalkers,self.Niter,self.Nparameters])
                self.chi2 = np.zeros([self.Nwalkers,self.Niter])
            else:
                pass
            
            self.data[i,:,:] = temp[:,1::]
            self.chi2[i,:] = temp[:,0]
            
        return
        
    def get_confidence_interval(self,interval=0.95,Tburn=0):
        
        Int_up = np.percentile(self.data[:,Tburn:,:],100*(0.5+interval/2.0),axis={0,1})
        Int_dn = np.percentile(self.data[:,Tburn:,:],100*(0.5-interval/2.0),axis={0,1})
        
        self.CI = np.zeros([len(Int_up),2])
        
        self.CI[:,0] = Int_up
        self.CI[:,1] = Int_dn
        
        return 
        
    def get_errorbars(self,interval=0.95,Tburn=0):
        
        self.get_max_likelihood()
        self.get_confidence_interval(interval=interval,Tburn=Tburn)
        
        self.errorup = np.zeros(len(self.best_params))
        self.errordn = np.zeros(len(self.best_params))
        
        self.errorup = self.CI[:,0] - self.best_params
        self.errordn = self.CI[:,1] - self.best_params
        
    def GelmanRubin(self, N=100,interval=0.95):
        '''
        Compute the Gelman Rubin diagnostic on the currently loaded chains.
        Calculates every Nth iteration on the last N/2 samples in the chain.
        '''        
        
        assert N % 2 ==0
        assert interval >=0
        assert interval <=1
        
        self.GR = np.zeros([len(range(N,self.Niter-N,N)),self.Nparameters])
        self.GR2 =np.zeros([len(range(N,self.Niter-N,N)),self.Nparameters])
        
        
        # first Gelman Rubin statistic, compares variances
        for i in range(N,self.Niter-N,N):
            
            # means for each chain
            m = self.Nwalkers
            n = i-i//2
            xj = np.mean(self.data[:,i//2:i,:],axis=1) 
            x = np.mean(self.data[:,i//2:i,:],axis=(0,1))
            W2 = np.zeros(self.Nparameters)
            
            
            
            # Within chain variance
            for j in range(self.Nparameters):
                W2[j] = 1.0/(m*(n-1))*np.sum(np.sum((self.data[:,i//2:i,j]-(xj[:,j])[:,np.newaxis])**2,axis=1),axis=0)
            W = 1/float(self.Nwalkers)*np.sum(np.std(self.data[:,i//2:i,:],axis=1)**2,axis=0)         
            
          
            # Between chain variance  
            B2 = n/float(m-1) * np.sum((xj-x)**2,axis=0)
            B = (i-i//2)/float(self.Nwalkers-1)*np.sum((np.mean(self.data[:,i//2:i,:],axis=1)-np.mean(np.mean(self.data[:,i//2:i,:],axis=1),axis=0))**2,axis=0)

            variance = (1-1/float(i-i//2))*W+B/float(i-i//2)
            v2 = (n-1)/float(n)*W2 +B2/float(n)
            
            #Reduction = np.sqrt(variance/W)
            R = (m+1)/float(m)*v2/W2 -(n-1)/float(m*n)
            
            self.GR[i//N-1,:] = R
            #self.GR[i//N-1,:] = R 
        
        # second Gelman Rubin.  Compares confidence intervals
        for i in range(N,self.Niter-N,N):
            Pup_m = np.percentile(self.data[:,:i,:],100*(0.5+interval/2.0),axis=1)
            Pdn_m = np.percentile(self.data[:,:i,:],100*(0.5-interval/2.0),axis=1)
            
            Pup_t = np.percentile(self.data[:,:i,:],100*(0.5+interval/2.0),axis={0,1})
            Pdn_t = np.percentile(self.data[:,:i,:],100*(0.5-interval/2.0),axis={0,1})
            
            Clm = Pup_m-Pdn_m
            Clt = Pup_t-Pdn_t
            

            self.GR2[i//N-1,:] = Clt/np.mean(Clm,axis=0)
            self.alpha = 1-interval
            
        

        return
        
    def Get_PostMean(self,Tburn):
        '''
        Calculate the posterior mean of the parameters, starting at 
        the Tburn-th iteration, and proceeding until the end.
        '''
        z = np.min(self.chi2)/2.0
        prob = np.exp(-self.chi2/2.0+z)
        prob /= np.sum(prob)
        
        self.Postmean = np.zeros(self.Nparameters)
        for i in range(self.Nparameters):
            self.Postmean[i] = np.sum(prob*self.data[:,:,i])
        
        return
        
    def cut_chains(self,Tburn):
        '''
        cut the burn in chains out of the chains used for study.
        '''        
        
        self.data = self.data[:,Tburn::,:]
        self.chi2 = self.chi2[:,Tburn::]
        self.Niter = self.data.shape[1]
        return        
        
    def cut_walker(self,walker_id):
        '''
        Remove walkers from the data used for study (risky)
        '''
        data = np.zeros([self.Nwalkers-1,self.Niter,self.Nparameters])
        chi2 = np.zeros([self.Nwalkers-1,self.Niter])
        for i in range(self.Nwalkers):
            if i < walker_id:
                data[i,:,:]   = self.data[i,:,:]
                chi2[i,:] = self.chi2[i,:]
            elif i>walker_id:
                data[i-1,:,:] = self.data[i,:,:]
                chi2[i-1,:] = self.chi2[i,:]
            else:
                pass
        self.data = data
        self.chi2 = chi2
        self.Nwalkers -= 1
        
        return
    def cut_parameter(self,parameter_id):
        data = np.zeros([self.Nwalkers,self.Niter,self.Nparameters-1])
        for i in range(self.Nparameters):
            if   i < parameter_id:
                data[:,:,i] = self.data[:,:,i]
            elif i > parameter_id:
                data[:,:,i-1] = self.data[:,:,i]
            else:
                pass
        self.data = data
        self.Nparameters -= 1
            
        return
        
    def get_max_likelihood(self):
        best_walker = np.argmin(np.min(self.chi2,axis=1))
        best_step = np.argmin(self.chi2[best_walker,:])
        self.best_chi2 = self.chi2[best_walker,best_step]
        self.best_params=self.data[best_walker,best_step,:]
        
        return
        
# ======================================================================

        
            