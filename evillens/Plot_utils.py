# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:35:36 2016

@author: warrenmorningstar
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import evillens as evil
import matplotlib.gridspec as gridspec
import struct
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.mlab import griddata
from operator import mul


def Plot_chains(MCMC,paramslist=None,Numpars=None,Tburn=None,Niter=None,figsize=[5,5], \
                Nrows=4, Ncols=4,PlotTitle=None, Filename=None):
    
    if paramslist is not None:
        paramslist = paramslist
    
    if Numpars is not None:
        Numpars = Numpars
    else:
        Numpars = MCMC.Nparameters
    
    if Tburn is not None:
        Tburn = Tburn
    else:
        Tburn = 0
        
    if Niter is not None:
        Niter = Niter
    else:
        Niter = MCMC.Niter
        
    assert Numpars <= Nrows*Ncols
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(Nrows,Ncols)
    gs.update(left=0.05,right=0.95,wspace=0.5,hspace=0.1)
    
    # Get Best Chi2 for later
    BestData = MCMC.data[np.argmin(MCMC.chi2,axis=0),range(Niter),:]
    BestChi2_i = np.min(MCMC.chi2,axis=0)
    BestChi2   = [np.argmin(BestChi2_i[:j]) for j in range(1,Niter-Tburn)]
    
    for i in range(Numpars):
        plt.subplot(gs[i//Ncols,i % Ncols])
        for j in range(MCMC.Nwalkers):
            plt.plot(MCMC.data[j,Tburn:Niter,i],linewidth=0.5,alpha=0.3)
        if paramslist is not None:
            plt.ylabel(paramslist[i])
        else:
            pass
        # Plot the average
        plt.plot(np.mean(MCMC.data[:,Tburn:Niter,i],axis=0),'k-',linewidth=1.0)    
        
        # Plot the best
        plt.plot(BestData[BestChi2,i],'r-')
        
        # Plot the posterior mean
        #z = np.min(MCMC.chi2,axis=0)/2.0
        #prob = np.zeros(MCMC.chi2.shape)
        #for j in range(MCMC.Niter):
        #    prob[:,j] = np.exp(-MCMC.chi2[:,j]/2.0+z[j])
        #    prob[:,j] /= np.sum(prob[:,j])
        #plt.plot(np.sum(MCMC.data[:,Tburn:Niter,i]*prob,axis=0),'g-',linewidth=0.5)
        
        
        # Labels and limits
        if i+Ncols >= Numpars:
            plt.xlabel('Number of iterations')
            plt.xticks(np.linspace(Tburn,Niter,5))
        else:
            plt.xticks(np.linspace(Tburn,Niter,5),[])
        
        plt.xlim(0,Niter-Tburn)
    
    if PlotTitle is not None:    
        plt.suptitle(PlotTitle, fontsize=22)

    # if in notebook, display, otherwise save to png file
    
    try:
        __IPYTHON__
        plt.show()
    except NameError:
        if Filename is not None:
            pngfile = Filename+'.png'
            plt.savefig(pngfile)
            print "Saved plot to "+pngfile
        else:
            print "Specify a filename to save plot"
            
            
    return fig
    
def Plot_chi2(MCMC,Tburn=0,Title=None,Filename=None,figsize=[5,5]):
    
    fig = plt.figure(figsize=figsize)
    
    if Tburn > MCMC.Niter:
        print('Burn Time is larger than number of iterations.  Setting to zero')
        Tburn = 0
    else:
        pass
    
    for i in range(MCMC.Nwalkers):
        plt.plot(MCMC.chi2[i,Tburn:],alpha=0.3)
    
    plt.xlabel('# of Iterations')
    plt.ylabel(r'$\mathcal{E}$')
    if Title is not None:    
        plt.title(Title)
    
    try:
        __IPYTHON__
        plt.show()
    except NameError:
        if Filename is not None:
            pngfile = Filename+'.png'
            plt.savefig(pngfile)
            print "Saved plot to "+pngfile
        else:
            print "Specify a filename to save plot"
            
    return fig

    
    
def Plot_dirty_image(Vis_data,u,v,Vis_model=None,Img_L = 4.0, Img_cent=[0,0]  \
                        ,Num_pixels = 100,figsize=[10,5],Title=None           \
                        ,fileneme=None,Flipped=False):
    '''
    Plot the dirty image from the data, the dirty image from the model, and
    the residuals (residuals in units of sigma)
    - Vis_data is filename of Visibility data
    - Vis_model is filename of Visibility model
    - u and v are filenames of uv coordinates
        
    '''
    with open(Vis_data, mode='rb') as file:
        fileContent = file.read()
        Vis_data = np.array(struct.unpack("d"*(len(fileContent)//8),fileContent))
    file.close()
    Vis_data = Vis_data[::2]+1j*Vis_data[1::2]
    
    if Vis_model is not None:
        Vis_model = np.loadtxt(Vis_model)
        Vis_model = np.array(Vis_model)
        Vis_model = Vis_model[::2]+1j*Vis_model[1::2]
        
    with open(u, mode='rb') as file:
        fileContent = file.read()
        u = np.array(struct.unpack("d"*(len(fileContent)//8),fileContent))
    file.close()
    with open(v, mode='rb') as file:
        fileContent = file.read()
        v = np.array(struct.unpack("d"*(len(fileContent)//8),fileContent))
    file.close()
        
        
    x = np.linspace(Img_cent[0]-Img_L/2.0,Img_cent[0]+Img_L/2.0,Num_pixels)
    y = np.linspace(Img_cent[1]-Img_L/2.0,Img_cent[1]+Img_L/2.0,Num_pixels)
        
    #convert to radians
    x /=(3600*180/np.pi)
    y /=(3600*180/np.pi)
    
    img_da = np.zeros([Num_pixels,Num_pixels])
    if Vis_model is not None:
        img_mo = np.zeros([Num_pixels,Num_pixels])
    DB = np.zeros([Num_pixels,Num_pixels])
        
       
    for i in range(Num_pixels):
        for j in range(Num_pixels):
            img_da[i,j] = 2*np.sum((Vis_data *np.exp(2j*np.pi*(u*x[j]+v*y[i]))).real)
            if Vis_model is not None:
                img_mo[i,j] = 2*np.sum((Vis_model*np.exp(2j*np.pi*(u*x[j]+v*y[i]))).real) 
            DB[i,j] = 2*np.sum(np.exp(2j*np.pi*(u*x[j]+v*y[i])).real)
    
    if Vis_model is not None:    
        resid = (img_da-img_mo)/np.std(img_da-img_mo)
        
    # convert x,y back to arcsec
    x*=(3600*180/np.pi)
    y*=(3600*180/np.pi)
    
    x,y = np.meshgrid(x,y)
    
    
    if Vis_model is not None:    
        fig = plt.figure(figsize=figsize)
        if Title is not None:
            plt.suptitle(Title,fontsize=22)
        gs = gridspec.GridSpec(1,3)
        gs.update(left=0.1,right=0.9,wspace=0.03)
        
        plt.subplot(gs[0,0])
        if Flipped is False:
            plt.imshow(img_da,origin='lower',extent=(np.min(x),np.max(x) \
                ,np.min(y),np.max(y)), cmap='RdBu_r'         \
                ,interpolation='nearest', vmin= np.min(img_da)   \
                ,vmax=np.max(img_da))
            plt.contour(x-(np.max(x)-np.mean(x))/1.2,y-(np.max(y)-np.mean(y))/1.2,DB,levels=[np.max(DB)/2.0],colors='k',linewidth=1.5)
            plt.xlim(np.min(x),np.max(x))
            plt.ylim(np.min(y),np.max(y))
        else:
            plt.imshow(img_da,origin='lower',extent=(np.min(x),np.max(x) \
                ,np.max(y),np.min(y)), cmap='RdBu_r'         \
                ,interpolation='nearest', vmin= np.min(img_da)   \
                ,vmax=np.max(img_da))
            plt.contour(x-(np.max(x)-np.mean(x))/1.2,y-(np.max(y)-np.mean(y))/1.2,np.flipud(DB),levels=[np.max(DB)/2.0],colors='k',linewidth=1.5)
            plt.ylim(np.min(y),np.max(y))
            plt.xlim(np.min(x),np.max(x))
        plt.title('Data')
        plt.ylabel('y (arcseconds)',fontsize=14)
        plt.xlabel('x (arcseconds)',fontsize=14)
        
        plt.subplot(gs[0,1])
        if Flipped is False:
            plt.imshow(img_mo,origin='lower',extent=(np.min(x),np.max(x) \
                ,np.min(y),np.max(y)), cmap='RdBu_r'          \
                ,interpolation='nearest', vmin= np.min(img_da)   \
                ,vmax=np.max(img_da))
        else:
            plt.imshow(img_mo,origin='lower',extent=(np.min(x),np.max(x) \
                ,np.max(y),np.min(y)), cmap='RdBu_r'          \
                ,interpolation='nearest', vmin= np.min(img_da)   \
                ,vmax=np.max(img_da))
            plt.ylim(np.min(y),np.max(y))
        plt.yticks([])
        plt.title('Model')
        plt.xlabel('x (arcseconds)',fontsize=14)
        
        plt.subplot(gs[0,2])
        if Flipped is False:
            plt.imshow(resid,origin='lower',extent=(np.min(x),np.max(x) \
                ,np.min(y),np.max(y)), cmap='RdBu_r'          \
                ,interpolation='nearest')
        else:
            plt.imshow(resid,origin='lower',extent=(np.min(x),np.max(x) \
                ,np.max(y),np.min(y)), cmap='RdBu_r'          \
                ,interpolation='nearest',vmin=-5,vmax=5)
            plt.ylim(np.min(y),np.max(y))
        plt.yticks([])
        plt.title('Residuals')
        plt.xlabel('x (arcseconds)',fontsize=14)
        
        cax = fig.add_axes([0.92,0.2,0.02,0.62])
        plt.colorbar(cax=cax,label=r'$\sigma$')
        
    else:
        fig = plt.figure(figsize=figsize)
        plt.imshow(img_da,origin='lower'*Flipped+'upper'*(1-Flipped),extent=(np.min(x),np.max(x) \
            ,np.min(y)*(1-Flipped)+np.max(y)*Flipped,np.max(y)*(1-Flipped)+np.min(y)*Flipped), cmap='viridis'         \
            ,interpolation='nearest', vmin= np.min(img_da)   \
            ,vmax=np.max(img_da))
        plt.colorbar()
        plt.contour(x-(np.max(x)-np.mean(x))/1.2,y-(np.max(y)-np.mean(y))/1.2,DB*(1-Flipped)+Flipped*np.flipud(DB),levels=[np.max(DB)/2.0],colors='k',linewidth=1.5)
        plt.xlim(np.min(x),np.max(x))
        plt.ylim(np.min(y),np.max(y))
        
    try:
        __IPYTHON__
        plt.show()
    except NameError:
        if Filename is not None:
            pngfile = Filename+'.png'
            plt.savefig(pngfile)
            print "Saved plot to "+pngfile
        else:
            print "Specify a filename to save plot"    
                
    return fig
        
def Plot_GR(MCMC,GR1 = True,figsize=[10,10],Nrows=4,Ncols=4,yrange=[0.002,1.2],    \
            paramslist=None, Title=None,Filename=None):
    '''
    Plot the Gelman Rubin statistic for the mcmc object.
    '''
        
    if MCMC.GR is None:
        MCMC.GelmanRubin()
    else:
        pass
        
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(Nrows,Ncols)
    gs.update(left=0.05, right=0.95, wspace=0.03,hspace=0.03)
    for i in range(MCMC.Nparameters):
        plt.subplot(gs[i//Ncols,i%Ncols])
        if GR1 is True:
            plt.plot(MCMC.GR[:,i]-1)
            plt.plot([0,len(MCMC.GR[:,i])],[0.01,0.01],'k--')
            plt.plot([0,len(MCMC.GR[:,i])],[0.1,0.1],'r--')
        else:
            plt.plot(MCMC.GR2[:,i]-1)
            plt.plot([0,len(MCMC.GR[:,i])],[MCMC.alpha,MCMC.alpha],'r--')
        plt.ylim(yrange[0],yrange[1])
        plt.yscale('log')

       
        if i+Ncols>=MCMC.Nparameters:
            plt.xticks(np.arange(0,len(MCMC.GR[:,i]),10))
        else:
            plt.xticks(np.arange(0,len(MCMC.GR[:,i]),10),[])
            
        if i%Ncols !=0:
            plt.yticks([0.01,0.1],[])
        else:
            plt.yticks([0.01,0.1])
            
        plt.xlim(0,len(MCMC.GR[:,i]))
            
        # clever way to make text labels coordinate invariant
        if paramslist is not None:
            assert len(paramslist) == MCMC.GR.shape[1]
                
            yext = np.log10(yrange[1]/yrange[0])
            plt.text(len(MCMC.GR[:,i])/2.0,10**(np.log10(yrange[1])-yext*0.1),paramslist[i])
    if Title is not None:
        plt.suptitle(Title)
            
    fig.text(0.00, 0.615, r'$R_{GR}-1$', rotation="vertical", va="center",fontsize=14)
    fig.text(0.45, 0.275, 'Check number', va="center",fontsize=14)
        
    try:
        __IPYTHON__
        plt.show()
    except NameError:
        if Filename is not None:
            pngfile = Filename+'.png'
            plt.savefig(pngfile)
            print "Saved plot to "+pngfile
        else:
            print "Specify a filename to save plot"
                
    return fig
    
    
    
def Plot_Triangle(MCMC,paramslist=None,Title=None,Filename=None,levels=[0.68,0.95],prange = None,truths=None,ticklabelsize=12,Nbins=None):
    '''
    Triangle plot of the parameters
    '''
    
    if paramslist is not None:
        paramslist = paramslist
    else:
        paramslist = ['' for i in range(MCMC.Nparameters)]
        
    if truths is not None:
        truths = truths
    else:
        truths = [ 10**10 for i in range(MCMC.Nparameters) ]
        
    if Nbins is None:
        # default number of bins determined by number of samples
        Nbins = int((MCMC.Niter*MCMC.Nwalkers)**(1/4.0))
    
    lkw = dict(fontsize=24)
    
    data = MCMC.data.reshape(-1,MCMC.Nparameters)
    fig = corner.corner(data,levels=levels,labels=paramslist,truths=truths, \
                        range=prange,plot_datapoints=False,bins=Nbins,fill_contours=True,max_n_ticks=3,label_kwargs=lkw)
    
    # adjust size of ticklabels
    for ax in fig.get_axes():
        ax.tick_params(labelsize=ticklabelsize)
    
    if Title is not None:
        plt.suptitle(Title,fontsize=30)
        
    
    try:
        __IPYTHON__
        plt.show()
    except NameError:
        if Filename is not None:
            pngfile = Filename+'.png'
            plt.savefig(pngfile)
            print "Saved plot to "+pngfile
        else:
            print "Specify a filename to save plot"
                
    return fig
    
def Plot_source(sourcedir,Npixels,source_L,sourcecent,includecaustics=True,\
                causticX=None,causticY=None,Ngridlines=None,SNR=False,SNRcut=3.0,srcerr=None,figsize=[10,10],Title=None,Filename=None,Flipped=False):
    src = np.loadtxt(sourcedir).reshape([Npixels,Npixels]).T
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1,2)
    gs.update(left=0.1,right=0.9,top =1.0,wspace=0.03)
    plt.subplot(gs[0,0])
    
    options = dict(interpolation='nearest',cmap='viridis',origin='lower')
    if Flipped is False:
        options['extent'] =(sourcecent[0]-source_L/2.0,sourcecent[0]+source_L/2.0,\
                        sourcecent[1]-source_L/2.0,sourcecent[1]+source_L/2.0)
    else:
        options['extent'] = (sourcecent[0]-source_L/2.0,sourcecent[0]+source_L/2.0,\
                        -sourcecent[1]+source_L/2.0,-sourcecent[1]-source_L/2.0)
    if SNR is True:
        srcerr = np.loadtxt(srcerr).reshape([Npixels,Npixels]).T
        SigNoise = src * np.sqrt(srcerr)
        src[np.where(SigNoise<SNRcut)] =  0.0
        options['vmin'] = 0.75*np.min(src[np.where(SigNoise>SNRcut)])
    
    plt.imshow(src, **options)
    
    plt.ylabel('y (arcsec)')
    plt.xlabel('x (arcsec)')
    if Title is not None:
        plt.suptitle(Title)
    
    if (includecaustics is True) and (causticX is not None):
        causticX = np.loadtxt(causticX)*3600*180/np.pi
        causticY = np.loadtxt(causticY)*3600*180/np.pi
        
        jxy,jxx = np.gradient(causticX)
        jyy,jyx = np.gradient(causticY)
        
        A = jxx*jyy - jxy*jyx
        
        
        if Flipped is False:
            plt.contour(causticX,causticY,A,levels=[0],colors='k',linewidths=2)
        else:
            plt.contour(causticX,-causticY,A,levels=[0],colors='k',linewidths=2)

            
    
    if Flipped is False:
        plt.ylim(-source_L/2.0+sourcecent[1],source_L/2.0+sourcecent[1])
        plt.xlim(-source_L/2.0+sourcecent[0],source_L/2.0+sourcecent[0])
        plt.yticks(np.linspace(-source_L/2.0,source_L/2.0,5)+sourcecent[1])
    else:
        plt.ylim(-source_L/2.0-sourcecent[1],source_L/2.0-sourcecent[1])
        print(-source_L/2.0-sourcecent[1],source_L/2.0-sourcecent[1])
        plt.xlim(-source_L/2.0+sourcecent[0],source_L/2.0+sourcecent[0])
        plt.yticks(np.linspace(-source_L/2.0,source_L/2.0,5)-sourcecent[1])
        
    
    plt.subplot(gs[0,1])
    if Flipped is False:
        plt.imshow(src,**options)
                        
        plt.yticks(np.linspace(-source_L/2.0,source_L/2.0,5)+sourcecent[1],[])
        plt.ylim(-source_L/2.0+sourcecent[1],source_L/2.0+sourcecent[1])
        plt.xlim(-source_L/2.0+sourcecent[0],source_L/2.0+sourcecent[0])
    else:
        plt.imshow(src,**options)
        plt.yticks(np.linspace(-source_L/2.0,source_L/2.0,5)+sourcecent[1],[])
        plt.ylim(-source_L/2.0-sourcecent[1],source_L/2.0-sourcecent[1])
        plt.xlim(-source_L/2.0+sourcecent[0],source_L/2.0+sourcecent[0]) 
    plt.xlabel('x (arcsec)')
    
    cax = fig.add_axes([0.92,0.2,0.02,0.72])
    plt.colorbar(cax=cax)
    
    try:
        __IPYTHON__
        plt.show()
    except NameError:
        if Filename is not None:
            pngfile = Filename+'.png'
            plt.savefig(pngfile)
            print "Saved plot to "+pngfile
        else:
            print "Specify a filename to save plot"
                
    return fig
    
    
def Compare_Triangles(MCMC1,MCMC2,paramslist=None,Title=None,Filename=None,levels=[0.68,0.95],truths=None,Nbins=20,show_both=False):
    
    d1 = MCMC1.data.reshape(-1,MCMC1.Nparameters)
    d2 = MCMC2.data.reshape(-1,MCMC2.Nparameters)
    
    if MCMC1.Nparameters != MCMC2.Nparameters:
        raise Exception('MCMC objects must have the same number of parameters \n')
    
    if paramslist is not None:
        paramslist=paramslist
    else:
        paramslist = ['' for i in range(MCMC1.Nparameters)]
    
    if truths is not None:
        truths = truths
    else:
        truths = [10**10 for i in range(MCMC1.Nparameters)]
        
    if Nbins is None:
        Nbins = 50
        
    # if we want to see both sets of contours, set the ranges manually
    if show_both is True:
        upper_lims = np.vstack([np.max(MCMC1.data,axis=(0,1)),np.max(MCMC2.data,axis=(0,1))])
        lower_lims = np.vstack([np.min(MCMC1.data,axis=(0,1)),np.min(MCMC2.data,axis=(0,1))])
        ranges = []
        for i in range(MCMC1.Nparameters):
            ranges.append((lower_lims[:,i].min(),upper_lims[:,i].max()))
        print ranges
    else:
        ranges=None
        
    fig = corner.corner(d1,range=ranges,levels = levels, labels=paramslist , truths=truths,plot_datapoints=False,bins=Nbins,fill_contours=True,hist_kwargs=dict(normed=True))
    corner.corner(d2,range=ranges,fig=fig,levels=levels,color='blue',plot_datapoints=False,bins=Nbins,fill_contours=True,hist_kwargs=dict(normed=True));
    
    if Title is not None:
        plt.suptitle(Title,fontsize=22)
        
    try:
        __IPYTHON__
        plt.show()
    except NameError:
        if Filename is not None:
            pngfile = Filename+'.png'
            plt.savefig(pngfile)
            print "Saved plot to "+pngfile
        else:
            print "Specify a filename to save plot"
                
    return fig
    
    
def Compare_chains(MCMClist,paramslist=None,Numpars=None,Tburn=None,Niter=None,figsize=[5,5], \
                Nrows=4, Ncols=4,PlotTitle=None, Filename=None):
                    
    if paramslist is not None:
        paramslist = paramslist
    
    if Numpars is not None:
        Numpars = Numpars
    else:
        Numpars = MCMClist[0].Nparameters
    
    if Tburn is not None:
        Tburn = Tburn
    else:
        Tburn = 0
        
    if Niter is not None:
        Niter2 = []
        for i in range(len(MCMClist)):
            Niter2.append(Niter)
            Niter = Niter2
    else:
        Niter = []
        for i in range(len(MCMClist)):
            Niter.append(MCMClist[i].Niter)
        
    assert Numpars <= Nrows*Ncols
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(Nrows,Ncols)
    gs.update(left=0.05,right=0.95,wspace=0.5,hspace=0.1)
    for i in range(Numpars):
        colorlist = ['k','r','b','c','m','g']
        plt.subplot(gs[i//Ncols,i % Ncols])
        for j in range(len(MCMClist)):
            for k in range(MCMClist[j].Nwalkers):
                plt.plot(MCMClist[j].data[k,Tburn:Niter[j],i],colorlist[j],alpha=0.3)
            
        if paramslist is not None:
            plt.ylabel(paramslist[i])
        else:
            pass
        if i+Ncols >= Numpars:
            plt.xlabel('Number of iterations')
            plt.xticks(np.linspace(Tburn,np.max(Niter),5))
        else:
            plt.xticks(np.linspace(Tburn,np.max(Niter),5),[])
        
        plt.xlim(Tburn,np.max(Niter))
    
    if PlotTitle is not None:    
        plt.suptitle(PlotTitle, fontsize=22)

    # if in notebook, display, otherwise save to png file
    
    try:
        __IPYTHON__
        plt.show()
    except NameError:
        if Filename is not None:
            pngfile = Filename+'.png'
            plt.savefig(pngfile)
            print "Saved plot to "+pngfile
        else:
            print "Specify a filename to save plot"
            
            
    return fig
    
    
def Plot_Subhalos(directory,SubFilesList,xlist=None,ylist=None,Ncols=3, \
        figsize=[10,10],Npixels=60,xlim=[-1,1],ylim=[-1,1],Flipped=False, \
        plotlabels=None,Title = None):
    
    # handling inputs
    SubList = SubFilesList
    Npannels = len(SubList)
    Ncols = Ncols
    Nrows = Npannels // Ncols +1*np.not_equal(0,Npannels % Ncols)
    
    
    # create figure object
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(Nrows,Ncols)
    gs.update(left=0.05,right=0.9,wspace=0.03,hspace=0.03)
    
    
    # set up plotting options.
    if Flipped is False:
        options = dict(origin='lower',extent=(xlim[0],xlim[1],ylim[0],ylim[1]))
    else:
        options = dict(origin='lower',extent=(xlim[0],xlim[1],ylim[1],ylim[0]))
    options['interpolation'] = 'Nearest'
    options['vmin']=-25
    options['vmax']=25
    clrmap = LinearSegmentedColormap.from_list('mycmap',['red', 'white','black'])
    options['cmap'] = clrmap
    
    
    
    for i in range(len(SubList)):
        img = np.loadtxt(directory+SubList[i])[1:,6].reshape([Npixels,Npixels])
        plt.subplot(gs[i//Ncols,i%Ncols])
        plt.imshow(img, **options)
        
        if i+Ncols >= len(SubList):
            plt.xticks([-3,-2,-1,0,1,2,3])
            plt.xlabel('x (arcsec)')
        else:
            plt.xticks([-3,-2,-1,0,1,2,3],[])
            
        if i%Ncols !=0:
            plt.yticks([-3,-2,-1,0,1,2,3],[])
        else:
            plt.yticks([-3,-2,-1,0,1,2,3])
            plt.ylabel('y (arcsec)')
            
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])
        if plotlabels is not None:
            plt.text((xlim[1]+xlim[0])/2.0,5*(ylim[1]-ylim[0])/12.0+(ylim[1]+ylim[0])/2.0,plotlabels[i],ha='center')
        
    
    
    cax = fig.add_axes([0.92,0.2,0.02,0.62])
    plt.colorbar(cax=cax,label=r'$\Delta\mathcal{E}$')
    
    if Title is not None:
        plt.suptitle(Title,fontsize=18)
    
    return fig
    
def Plot_All( visdatadir , vismod , srcmod , imgmod , src_L , Npix_s , imL  , Npix_L , Npix_img \
                , src_cent=[0,0] , img_cent =[0,0] , dirty_image_size = None, Flipped=True , figscale=1  \
                , caustics=False , Causticx = None, Causticy=None , SNR=False , SNRcut=3.0 , srcerr=None,title=None):
    '''
    A plot that includes the dirty map, predicted map, residuals, lensmodel, and source reconstruction.
    - visdatadir is directory of visibility data (assumes filename is vis_chan_n.bin)
    - vismod is visibility model file
    - srcmod is source model file
    - imgmod is the image model file
    - src_L is length of source grid side
    - Npix_s is number of source pixels per side
    - imL is length of image on a size
    - Npix_L is number of lens pixels per side
    - Npix_img is the number of image pixels
    - Flipped specifies if figure is flipped
    - figsize specifies figure size (in inches)
    - caustics is a flag for including caustics in the source plot
    - Causticx and Causticy are the files for the source caustics
    - Ngridlines is number of caustic gridlines
    - SNR is to include a SNR cut
    - SNRcut is value of SNR cut
    - srcerr is list of source pixel errors (to get SNR)
    '''
    
    ## first load the files 
    
    # visibility data
    Vis_data = evil.load_binary(visdatadir+'vis_chan_0.bin')
    Vis_data = Vis_data[::2]+1j*Vis_data[1::2]

    # visibility model
    Vis_model = np.loadtxt(vismod)
    Vis_model = np.array(Vis_model)
    Vis_model = Vis_model[::2]+1j*Vis_model[1::2]
    
    # uv coordinates
    u = evil.load_binary(visdatadir+'u.bin')
    v = evil.load_binary(visdatadir+'v.bin')
    
    # Source and lens models
    src = np.loadtxt(srcmod).reshape([Npix_s,Npix_s]).T
    img = np.loadtxt(imgmod).reshape([Npix_img,Npix_img]).T
    
    ## Create dirty image, model, and residuals
    
    # Set up pixel grids
    if dirty_image_size is None:
        DIS = imL
        x = np.linspace(img_cent[0]-imL/2.0,img_cent[0]+imL/2.0,Npix_L)
        y = np.linspace(img_cent[1]-imL/2.0,img_cent[1]+imL/2.0,Npix_L)
    else:
        DIS = dirty_image_size
        x = np.linspace(img_cent[0]-DIS/2.0,img_cent[0]+DIS/2.0,Npix_L)
        y = np.linspace(img_cent[1]-DIS/2.0,img_cent[1]+DIS/2.0,Npix_L)
        
    # convert to radians
    x /=(3600*180/np.pi)
    y /=(3600*180/np.pi)
    
    # create images and Dirty Beam
    img_da = np.zeros([Npix_L,Npix_L])
    img_mo = np.zeros([Npix_L,Npix_L])
    DBeam  = np.zeros([Npix_L,Npix_L])
        
       
    for i in range(Npix_L):
        for j in range(Npix_L):
            img_da[i,j] = 2*np.sum((Vis_data *np.exp(2j*np.pi*(u*x[j]+v*y[i]))).real)
            img_mo[i,j] = 2*np.sum((Vis_model*np.exp(2j*np.pi*(u*x[j]+v*y[i]))).real) 
            DBeam[i,j] = 2*np.sum((np.exp(2j*np.pi*(u*(x[j]-np.mean(x))+v*(y[i]-np.mean(y))))).real) 
        
    resid = (img_da-img_mo)/np.std(img_da-img_mo)
        
    # convert x,y back to arcsec
    x*=(3600*180/np.pi)
    y*=(3600*180/np.pi)
    
    # Load Caustics if needed
    if caustics is True:
        cx = np.loadtxt(Causticx)*3600*180/np.pi
        cy = np.loadtxt(Causticy)*3600*180/np.pi
        jxy,jxx = np.gradient(cx)
        jyy,jyx = np.gradient(cy)
        A = jxx*jyy - jyx*jxy
    
    # setup figure
    
    fig = plt.figure(figsize=[27.5/4.0*figscale,8.5/4.0*figscale])
    gs = gridspec.GridSpec(1,5)
    gs.update(left=0.05,right=0.95,bottom=0.075,top=0.99,wspace=0.03)
    
    
    ## First figure panel
    
    # setup plotting options
    options = dict(interpolation='nearest')
    options['cmap']   = 'viridis'
    options['origin'] = 'lower'
    options['vmin'] = np.min(img_da)
    options['vmax'] = np.max(img_da)
    if Flipped is False:
        options['extent'] = (np.min(x),np.max(x),np.min(y),np.max(y))
    else:
        options['extent'] = (np.min(x),np.max(x),np.max(y),np.min(y))
        DBeam = np.flipud(DBeam)
        
    # create figure
    ax1 = plt.subplot(gs[0,0])
    dirty_map = plt.imshow(img_da,**options)
    plt.contour(x-(DIS)/2.6,y-(DIS)/2.6,DBeam,levels=[np.max(DBeam)/2.0],colors='k',linewidth=1.5)
    plt.xlim(np.min(x),np.max(x))
    plt.ylim(np.min(y),np.max(y))
    plt.xticks([])
    plt.yticks([])
    ax1.text((np.min(x)+np.max(x))/2.0,(np.min(y)+np.max(y))/2.0+(np.max(y)-np.min(y))/2.5,'Dirty Map',fontsize=6.0*figscale,bbox={'color':'white','alpha':0.5},horizontalalignment='center',verticalalignment='center')
    cax1 = fig.add_axes([0.05,0.17,0.175,0.05])
    cbar1 = plt.colorbar(dirty_map,cax=cax1,label='Flux (Jy / str)',orientation='horizontal')
    cbar1.set_label('Flux (Jy / Beam)',fontsize=6.0*figscale)
    ## Second figure panel
    
    # Same plotting options as first panel
    # create figure
    ax2 = plt.subplot(gs[0,1])
    predicted_map = plt.imshow(img_mo,**options)
    plt.xlim(np.min(x),np.max(x))
    plt.ylim(np.min(y),np.max(y))
    ax2.text((np.min(x)+np.max(x))/2.0,(np.min(y)+np.max(y))/2.0+(np.max(y)-np.min(y))/2.5,'Predicted Map',fontsize=6.0*figscale,bbox={'color':'white','alpha':0.5},horizontalalignment='center',verticalalignment='center')
    plt.yticks([])
    plt.xticks([])
    cax2 = fig.add_axes([0.23125,0.17,0.175,0.05])
    cbar2 = plt.colorbar(predicted_map,cax=cax2,label='Flux (Jy / str)',orientation='horizontal')
    cbar2.set_label('Flux (Jy / Beam)',fontsize=6.0*figscale)
    ## Third figure panel
    
    # Same plotting options as first panel
    # create figure
    ax3 = plt.subplot(gs[0,2])
    residual = plt.imshow(img_da-img_mo,**options)
    plt.contour(x,np.flipud(y),(img_da-img_mo)/np.std(img_da-img_mo),colors='r',levels=[2,4,6],linestyles='-')
    plt.contour(x,np.flipud(y),(img_da-img_mo)/np.std(img_da-img_mo),colors='r',levels=[-6,-4,-2],linestyles='--')
    plt.xlim(np.min(x),np.max(x))
    plt.ylim(np.min(y),np.max(y))
    ax3.text((np.min(x)+np.max(x))/2.0,(np.min(y)+np.max(y))/2.0+(np.max(y)-np.min(y))/2.5,'Residuals',fontsize=6.0*figscale,bbox={'color':'white','alpha':0.5},horizontalalignment='center',verticalalignment='center')
    plt.yticks([])
    plt.xticks([])
    cax3 = fig.add_axes([0.4125,0.17,0.175,0.05])
    cbar3 = plt.colorbar(residual,cax=cax3,label='Flux (Jy / str)',orientation='horizontal')
    cbar3.set_label('Flux (Jy / Beam)',fontsize=6.0*figscale)
    
    ## Fourth figure panel
    
    # new plotting options
    # setup plotting options
    options = dict(interpolation='nearest')
    options['cmap']   = 'viridis'
    options['origin'] = 'lower'
    options['vmin'] = np.min(img)
    options['vmax'] = np.max(img)
    if Flipped is False:
        options['extent'] = (img_cent[0]-imL/2.0,img_cent[0]+imL/2.0,img_cent[1]-imL/2.0,img_cent[1]+imL/2.0)
        ymin = img_cent[1]-imL/2.0
        ymax = img_cent[1]+imL/2.0
    else:
        options['extent'] = (img_cent[0]-imL/2.0,img_cent[0]+imL/2.0,-img_cent[1]+imL/2.0,-img_cent[1]-imL/2.0)
        ymin = -img_cent[1]-imL/2.0
        ymax = -img_cent[1]+imL/2.0
        
    # create figure
    ax4 = plt.subplot(gs[0,3])
    Skymodel = plt.imshow(img,**options)
    if Flipped is False:
        if caustics is True:
            plt.contour(cx,cy,A,levels=[0],colors='k',linewidths=2)
    else:
        if caustics is True:
            plt.contour(cx,-cy,A,levels=[0],colors='k',linewidths=2)
        
    plt.xlim(img_cent[0]-imL/2.0,img_cent[0]+imL/2.0)
    plt.ylim(ymin,ymax)
    ax4.text(img_cent[0],(ymin+ymax)/2.0+(ymax-ymin)/2.5,'Skymodel',fontsize=6.0*figscale,bbox={'color':'white','alpha':0.5},horizontalalignment='center',verticalalignment='center')
    ax4.set_yticklabels([])
    plt.yticks([])
    plt.xticks([])
    cax4 = fig.add_axes([0.59375,0.17,0.175,0.05])
    cbar4 = plt.colorbar(Skymodel,cax=cax4,label='Flux (Jy / str)',orientation='horizontal')
    cbar4.set_label('Flux (Jy / str)',fontsize=6.0*figscale)
    cbar4.ax.set_xticklabels(cbar4.ax.get_xticklabels(),rotation=45)
    # Fifth figure panel
    
    # new plotting options
    options = dict(interpolation='nearest')
    options['cmap'] = 'viridis'
    options['origin'] = 'lower'
    options['vmin'] = np.min(img)
    options['vmax'] = np.max(img)
    if Flipped is False:
        options['extent'] = (src_cent[0]-src_L/2.0,src_cent[0]+src_L/2.0,src_cent[1]-src_L/2.0,src_cent[1]+src_L/2.0)
        ymin = src_cent[1]-src_L/2.0
        ymax = src_cent[1]+src_L/2.0
    else:
        options['extent'] = (src_cent[0]-src_L/2.0,src_cent[0]+src_L/2.0,-src_cent[1]+src_L/2.0,-src_cent[1]-src_L/2.0)
        ymin = -src_cent[1]-src_L/2.0
        ymax = -src_cent[1]+src_L/2.0
        
    # create figure
    ax5 = plt.subplot(gs[0,4])
    srcplot = plt.imshow(src,**options)
    if caustics is True:
        if Flipped is False:
            plt.contour(cx,cy,A,levels=[0],colors='k',linewidths=2)
        else:
            plt.contour(cx,-cy,A,levels=[0],colors='k',linewidths=2)
        
    plt.xlim(src_cent[0]-src_L/2.0,src_cent[0]+src_L/2.0)
    plt.ylim(ymin,ymax)
    ax5.text(src_cent[0],(ymin+ymax)/2.0+(ymax-ymin)/2.5,'Reconstructed Source',fontsize=6.0*figscale,bbox={'color':'white','alpha':0.5},horizontalalignment='center',verticalalignment='center')
    ax5.set_yticklabels([])
    plt.yticks([])
    plt.xticks([])
    cax5 = fig.add_axes([0.775,0.17,0.175,0.05])
    cbar5 = plt.colorbar(srcplot,cax=cax5,label='Flux (Jy / str)',orientation='horizontal')
    cbar5.set_label('Flux (Jy / str)',fontsize=6.0*figscale)
    cbar5.ax.set_xticklabels(cbar5.ax.get_xticklabels(),rotation=45)
    # add colorbar
    #cax = fig.add_axes([0.92,0.2,0.02,0.62])
    #cbar = plt.colorbar(srcplot,cax=cax,label='Flux (Jy / str)')
    #cbar.set_label('Flux (Jy / str)',fontsize=6.0*figscale)
    
    ax3.text((np.min(x)+np.max(x))/2.0,(np.min(y)+np.max(y))/2.0+(np.max(y)-np.min(y))/1.75,title,fontsize=6.0*figscale,horizontalalignment='center',verticalalignment='center')
    
    
    return fig 
    
def Plot_Tesellated_Subs(subfileslist,xlim=None,ylim=None,Flipped=False,figscale=1,Num_levels=10,labels=None,sigcontours = False,Figtitle=None):
    
    num_panels = len(subfileslist)
    
    fig = plt.figure(figsize=[num_panels*5*figscale,5*figscale])
    plt.axes().set_aspect('equal','datalim')
    gs = gridspec.GridSpec(1,num_panels)
    gs.update(left=0.1,right=0.85,bottom=0.15,top=0.9,wspace=0.03)
    clrmap = LinearSegmentedColormap.from_list('mycmap',['red', 'white','black'])
    
    for i in range(num_panels):
        
        subdata = np.loadtxt(subfileslist[i])
        subdata = subdata[1:,:]
        subx = subdata[:,2]
        suby = subdata[:,3]
        dx2 = subdata[:,6]
        Nsubs = len(dx2)
        
        # Begin the tesellation
        T1_cent = np.array([subx[0],suby[0]])
        T1_size = np.linalg.norm(np.array([subx[1],suby[1]])-T1_cent)
        tesl = np.zeros([Nsubs,3,2])
        
        for j in range(Num_levels):
            
            A = tesl[:,0,0]*tesl[:,1,1]+tesl[:,0,1]*tesl[:,2,0]+tesl[:,1,0]*tesl[:,2,1]
            A -=tesl[:,0,0]*tesl[:,2,1]+tesl[:,0,1]*tesl[:,1,0]+tesl[:,1,1]*tesl[:,2,0]
            if np.sum(A==0) ==0:
                break
            
            for k in range(Nsubs):
                Ti_cent = np.array([subx[k],suby[k]])
                Ti_size = T1_size/(2.0**j)
                
                # Try forward facing triangle
                vert1 = Ti_cent+Ti_size*np.array([np.cos(0),np.sin(0)])
                vert2 = Ti_cent+Ti_size*np.array([np.cos(2*np.pi/3.0),np.sin(2*np.pi/3.0)])
                vert3 = Ti_cent+Ti_size*np.array([np.cos(4*np.pi/3.0),np.sin(4*np.pi/3.0)])
                
                B1 = (subx-vert1[0]) * (vert1[1]-vert2[1]) / (vert1[0]-vert2[0]) + Ti_cent[1]
                B2 = vert2[0]
                B3 = (subx-vert1[0]) * (vert3[1]-vert1[1]) / (vert3[0]-vert1[0]) + Ti_cent[1]
                
                if (np.sum((suby<B1+10**-5) & (subx>B2-10**-5) & (suby>B3-10**-5))==1) & (tesl[k,0,0] ==0):
                    tesl[k,0,:] = vert1
                    tesl[k,1,:] = vert2
                    tesl[k,2,:] = vert3
                
                # Try backward facing triangle
                vert1 = Ti_cent+Ti_size*np.array([np.cos(np.pi),np.sin(np.pi)])
                vert2 = Ti_cent+Ti_size*np.array([np.cos(2*np.pi/3.0+np.pi),np.sin(2*np.pi/3.0+np.pi)])
                vert3 = Ti_cent+Ti_size*np.array([np.cos(4*np.pi/3.0+np.pi),np.sin(4*np.pi/3.0+np.pi)])
                
                B1 = (subx-vert1[0]) * (vert1[1]-vert2[1]) / (vert1[0]-vert2[0]) + Ti_cent[1]
                B2 = vert2[0]
                B3 = (subx-vert1[0]) * (vert3[1]-vert1[1]) / (vert3[0]-vert1[0]) + Ti_cent[1]
                
                if (np.sum( (suby>B1-10**-5) & (subx<B2+10**-5) & (suby<B3+10**-5)) ==1) & (tesl[k,0,0] == 0):
                    tesl[k,0,:] = vert1
                    tesl[k,1,:] = vert2
                    tesl[k,2,:] = vert3
        
        # Reshape the constructed tesellation and assign sequential indices for tripcolor
        tesl2 = tesl.reshape(-1,2)
        inds  = np.arange(tesl2.shape[0]).reshape(tesl.shape[0],3)
        
        # Now have all that is needed for plotting tripcolor
        plt.subplot(gs[0,i])
        if (i==num_panels-1):
            subim = plt.tripcolor(tesl2[:,0],(1-2*Flipped)*tesl2[:,1],inds,dx2,vmin=-25,vmax=25,cmap=clrmap)
        else:
            plt.tripcolor(tesl2[:,0],(1-2*Flipped)*tesl2[:,1],inds,dx2,vmin=-25,vmax=25,cmap=clrmap)
        if i !=0:
            plt.yticks(np.arange(int(ylim[0]),int(ylim[1])+1,1),[])
        else:
            plt.yticks(np.arange(int(ylim[0]),int(ylim[1])+1,1),fontsize=18*figscale)
            plt.ylabel('y (arcsec)',fontsize=24*figscale)
        plt.xlabel('x (arcsec)',fontsize=24*figscale)
        plt.xticks(np.arange(int(xlim[0]),int(xlim[1])+1,1),fontsize=18*figscale)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if labels is not None:
            plt.text((xlim[0]+xlim[1])/2.0,(ylim[0]+ylim[1])/2.0+(ylim[1]-ylim[0])/2.5,labels[i],fontsize=24.0*figscale,horizontalalignment='center',verticalalignment='center',bbox={'color':'white','alpha':0.5})
        if sigcontours is True:
            xi,yi = np.meshgrid(np.linspace(xlim[0],xlim[1],50),np.linspace(ylim[0],ylim[1],50))
            zi = griddata(subx,(1-2*Flipped)*(suby),dx2/np.sqrt(abs(dx2)),xi,yi,interp='linear')
            plt.contour(xi,yi,zi,levels=[3,4,5],colors='b')
            plt.contour(xi,yi,zi,levels=[-5,-4,-3],colors='b',linestyles='--')
            
        
    cax = fig.add_axes([0.855,0.15,0.02,0.75])
    cbar = plt.colorbar(subim,cax=cax,label=r'$\Delta\mathcal{E}$')
    cbar.set_label(r'$\Delta\mathcal{E}$',fontsize=24*figscale)
    cbar.ax.tick_params(labelsize=18*figscale)
    if Figtitle is not None:
        plt.suptitle(Figtitle,fontsize=24*figscale)
            
    return fig
    
    
def Plot_Subhalo_Mass_Function(fileslist,mass_coords=None):
    '''
    Using linearized delta_chi2 maps, calculate and plot the subhalo mass function.
    
    Takes:
    
    - fileslist:  The list of filenames of the linearized maps.
    '''
    
    if mass_coords is None:
        dM = 10**8.5-10**8.0*np.ones(len(fileslist))
    else:
        dM = np.diff(mass_coords)
        dM = np.append(dM,dM[-1])
    
    # Arrays for subhalo pdfs and upper/lower limits
    Psubs = np.zeros([len(fileslist),100000])
    Upperlim = np.zeros(len(fileslist))
    Lowerlim = np.zeros(len(fileslist))
    
    # iterate over list of files
    for i in range(len(fileslist)):
        
        # Load subhalo map files
        subdat = np.loadtxt(fileslist[i])
        x  = subdat[1:,2]
        y  = subdat[1:,3]
        M  = subdat[1, 1]
        dE = subdat[1:,6]
        
        # get tesellation vertex coordinates and indices
        tesl, inds = reconstruct_subhalo_tesselation(x,y)
        
        # Get areas of cells
        Area =  tesl[:,0,0]*(tesl[:,1,1]-tesl[:,2,1])
        Area += tesl[:,1,0]*(tesl[:,2,1]-tesl[:,0,1])
        Area += tesl[:,2,0]*(tesl[:,0,1]-tesl[:,1,1])
        Area /= 2.0
        
        n_points=100000
        dN_dM = np.linspace(0,300,n_points)/1.0e8
        
        # ignore points with delta chi2<-5
        min_chi2 = -5.
        A  = Area[(dE>min_chi2)]
        dx2 = dE[(dE>min_chi2)]
        
        # Calculate PDF for one subhalo based on exclusions
        Pdf1 = np.exp(dN_dM * dM[i] *np.sum(A*(np.exp(-0.5*dx2)-1.)))
        
        # Calculate PDF based on detections (if any)
        dx2 = dE[(dE<min_chi2)]
        A = Area[(dE<min_chi2)]
        
        Pdf2 = np.zeros(len(dN_dM))
        try:
            for j in range(len(dN_dM)):
                Pdf2[j] = np.exp(-dN_dM[j] * dM[i] * np.sum(A))*reduce(mul,(1+dN_dM[j]*dM[i]*A*np.exp(-0.5*dx2)))
        except TypeError:
            print "no pixels have negative delta-chi2."
            Pdf2 += 1.
        Pdf3 = 1.
        
        pdf = Pdf1*Pdf2*Pdf3
        pdf /= np.sum(pdf*(dN_dM[1]-dN_dM[0]))
        Psubs[i,:] = pdf/np.max(pdf)
        indmax = np.argmax(pdf)
        
        CDF = np.cumsum(pdf*(dN_dM[1]-dN_dM[0]))
        Upperlim[i] = dN_dM[(CDF>0.95)][0]
    
    return Upperlim,Psubs,Pdf1,Pdf2,Pdf3, Area,dE
    
def Plot_Fisher_Forecast(fisher_matrix,params,param_IDs,figscale=1,param_names=None):
    
    if isinstance(fisher_matrix,basestring) is True:
        try:
            F = np.loadtxt(fisher_matrix)
        except(ValueError):
            F = evil.load_binary(fisher_matrix)
    elif len(fisher_matrix.shape) ==2:
        F = np.array(fisher_matrix)
        assert len(F.shape) == 2
        assert F.shape[0] == F.shape[1]
        
    Row,Col = np.meshgrid(param_IDs,param_IDs)
    F_reduced = F[Row,Col]
    num_panels = len(param_IDs)
    
    fig = plt.figure(figsize=[num_panels*figscale,num_panels*figscale])
    gs = gridspec.GridSpec(num_panels,num_panels)
    gs.update(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0,hspace=0)
    
    for i in range(num_panels):
        for j in range(i+1):
            if i!=j:
                print(i,j)
                # create and plot ellipse
                y,x = np.meshgrid(np.linspace(params[param_IDs[i]]-3.2*np.sqrt(F_reduced[i,i]),params[param_IDs[i]]+3.2*np.sqrt(F_reduced[i,i]), 100),np.linspace(params[param_IDs[j]]-3.2*np.sqrt(F_reduced[j,j]),params[param_IDs[j]]+3.2*np.sqrt(F_reduced[j,j]),100))
                cov = np.zeros([2,2])
                cov[0,0] = F_reduced[j,j]
                cov[1,1] = F_reduced[i,i]
                cov[1,0] = F_reduced[i,j]
                cov[0,1] = F_reduced[j,i]
                
                rho = cov[0,1]/np.sqrt(cov[0,0])/np.sqrt(cov[1,1])
                mu_x = params[param_IDs[j]]
                mu_y = params[param_IDs[i]]
                
                Prob = np.exp(-0.5/(1-rho**2)*(-2*rho*((x-mu_x)*(y-mu_y)/(np.sqrt(cov[0,0])*np.sqrt(cov[1,1])))+(x-mu_x)**2/cov[0,0]+(y-mu_y)**2/cov[1,1]))
                #Prob /= 2*np.pi*np.sqrt(cov[0,0])*np.sqrt(cov[1,1])*np.sqrt(1-rho**2)
                
                plt.subplot(gs[i,j])
                plt.contourf(x,y,Prob,levels=[0,np.exp(-0.5*9),np.exp(-0.5*4),np.exp(-0.5),1],cmap='gray_r')
                
                
                
                
            else:
                # create and plot 1d PDF
                x = np.linspace(params[param_IDs[j]]-3.2*np.sqrt(F_reduced[j,j]),params[param_IDs[j]]+3.2*np.sqrt(F_reduced[j,j]), 100)
                y = np.exp(-0.5*(x-params[param_IDs[j]])/F_reduced[j,j]*(x-params[param_IDs[j]]))/np.sqrt(2*np.pi*F_reduced[j,j])
                plt.subplot(gs[i,j])
                plt.plot(x,y,'k-')
                plt.xlim(np.min(x),np.max(x))
            if i+1 == num_panels:
                plt.xlabel(param_names[j],fontsize=24*figscale)
            if j == 0:
                plt.ylabel(param_names[i],fontsize=24*figscale)
    
        
def Compare_Forecasts(Fisherlist,params,param_IDs,MCMC=None,figscale=1,mean_subtracted=False,PlotLabels =None,Paramlabels=None):
    
    
    clist = ['Greys','Blues','Reds','Greens','Purples']
    clist2= ['k-','b-','r-','g-','m-']
    clist3= ['k','b','r','g','m']
    
    
    # axis limits array
    limits = np.zeros([len(param_IDs),len(param_IDs),4])
    labels = np.zeros([len(param_IDs),len(param_IDs),4])
    
    for k in range(len(Fisherlist)):
    
        if isinstance(Fisherlist[k],basestring) is True:
            try:
                F = np.loadtxt(Fisherlist[k])
            except(ValueError):
                F = evil.load_binary(Fisherlist[k])
            print F.shape
            F = F[1:].reshape([len(params),len(params)])
        elif len(Fisherlist[k].shape) ==2:
            F = np.array(Fisherlist[k])
            assert len(F.shape) == 2
            assert F.shape[0] == F.shape[1]
        
        Row,Col = np.meshgrid(param_IDs,param_IDs)
        F_reduced = F[Row,Col]
        num_panels = len(param_IDs)
        
        # create figure
        if k ==0:
            fig = plt.figure(figsize=[num_panels*figscale,num_panels*figscale])
            gs = gridspec.GridSpec(num_panels,num_panels)
            gs.update(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0,hspace=0)
            
            # Set axis limits and tick labels
            for i in range(num_panels):
                for j in range(i+1):
                    limits[i,j,0] = -3.2*np.sqrt(F_reduced[i,i])+ params[param_IDs[i]]
                    limits[i,j,1] = 3.2*np.sqrt(F_reduced[i,i]) + params[param_IDs[i]]
                    limits[i,j,2] = -3.2*np.sqrt(F_reduced[j,j])+ params[param_IDs[j]]
                    limits[i,j,3] = 3.2*np.sqrt(F_reduced[j,j]) + params[param_IDs[j]]
                    
                    labels[i,j,0] = -2*np.sqrt(F_reduced[i,i])+ params[param_IDs[i]]
                    labels[i,j,1] = 2*np.sqrt(F_reduced[i,i]) + params[param_IDs[i]]
                    labels[i,j,2] = -2*np.sqrt(F_reduced[j,j])+ params[param_IDs[j]]
                    labels[i,j,3] = 2*np.sqrt(F_reduced[j,j]) + params[param_IDs[j]]
                    
                    
                    
        else: # Adjust axis limits and tick labels if necessary
            for i in range(num_panels):
                for j in range(i+1):
                    limits[i,j,0] = np.min([limits[i,j,0],-3.2*np.sqrt(F_reduced[i,i])+ params[param_IDs[i]]])
                    limits[i,j,1] = np.max([limits[i,j,1],3.2*np.sqrt(F_reduced[i,i]) + params[param_IDs[i]]])
                    limits[i,j,2] = np.min([limits[i,j,2],-3.2*np.sqrt(F_reduced[j,j])+ params[param_IDs[j]]])
                    limits[i,j,3] = np.max([limits[i,j,3],3.2*np.sqrt(F_reduced[j,j]) + params[param_IDs[j]]])
                    
                    labels[i,j,0] = np.min([labels[i,j,0],-2*np.sqrt(F_reduced[i,i])+ params[param_IDs[i]]])
                    labels[i,j,1] = np.max([labels[i,j,1],2*np.sqrt(F_reduced[i,i]) + params[param_IDs[i]]])
                    labels[i,j,2] = np.min([labels[i,j,2],-2*np.sqrt(F_reduced[j,j])+ params[param_IDs[j]]])
                    labels[i,j,3] = np.max([labels[i,j,3],2*np.sqrt(F_reduced[j,j]) + params[param_IDs[j]]])
                    
    
        for i in range(num_panels):
            for j in range(i+1):
                if i!=j:

                    # create and plot ellipse
                    y,x = np.meshgrid(np.linspace(params[param_IDs[i]]-3.2*np.sqrt(F_reduced[i,i]),params[param_IDs[i]]+3.2*np.sqrt(F_reduced[i,i]), 100),np.linspace(params[param_IDs[j]]-3.2*np.sqrt(F_reduced[j,j]),params[param_IDs[j]]+3.2*np.sqrt(F_reduced[j,j]),100))
                    cov = np.zeros([2,2])
                    cov[0,0] = F_reduced[j,j]
                    cov[1,1] = F_reduced[i,i]
                    cov[1,0] = F_reduced[i,j]
                    cov[0,1] = F_reduced[j,i]
                
                    rho = cov[0,1]/np.sqrt(cov[0,0])/np.sqrt(cov[1,1])
                    mu_x = params[param_IDs[j]]
                    mu_y = params[param_IDs[i]]
                
                    Prob = np.exp(-0.5/(1-rho**2)*(-2*rho*((x-mu_x)*(y-mu_y)/(np.sqrt(cov[0,0])*np.sqrt(cov[1,1])))+(x-mu_x)**2/cov[0,0]+(y-mu_y)**2/cov[1,1]))
                    #Prob /= 2*np.pi*np.sqrt(cov[0,0])*np.sqrt(cov[1,1])*np.sqrt(1-rho**2)
                   
                    plt.subplot(gs[i,j])
                    plt.contourf(x,y,Prob,levels=[np.exp(-0.5*9),np.exp(-0.5*4),np.exp(-0.5),1],cmap=clist[k],alpha=0.9)
                    plt.contour(x,y,Prob,levels=[np.exp(-0.5*9),np.exp(-0.5*4),np.exp(-0.5),1],colors=[clist3[k]],linewidths=0.5)
                    
                    # overplot MCMC?
                    if (k==0) and (MCMC is not None) and (isinstance(MCMC,evil.MCMC)):
                        xdata = MCMC.data[:,:,param_IDs[j]].ravel()+mean_subtracted*(-np.mean(MCMC.data[:,:,param_IDs[j]])+params[param_IDs[j]])
                        ydata = MCMC.data[:,:,param_IDs[i]].ravel()+mean_subtracted*(-np.mean(MCMC.data[:,:,param_IDs[i]])+params[param_IDs[i]])
                        corner.hist2d(xdata,ydata,bins=50,levels=[0.68,0.95],plot_datapoints=False, plot_density=False,fill_contours=True);
                        
                    # adjust panel size if need be
                    xlabels = np.linspace(labels[i,j,2],labels[i,j,3],3)
                    ylabels = np.linspace(labels[i,j,0],labels[i,j,1],3)
                    
                    
                    if i+1==num_panels:
                        plt.xticks(xlabels,rotation=45)
                        if Paramlabels is not None:
                            plt.xlabel(Paramlabels[param_IDs[j]],fontsize=10*figscale)
                    else:
                        plt.xticks(xlabels,[])
                    if j==0:
                        plt.yticks(ylabels)
                        if Paramlabels is not None:
                            plt.ylabel(Paramlabels[param_IDs[i]],fontsize=10*figscale)
                    else:
                        plt.yticks(ylabels,[])
                    
                    plt.xlim(limits[i,j,2],limits[i,j,3])
                    plt.ylim(limits[i,j,0],limits[i,j,1])
                
                else:
                    # create and plot 1d PDF
                    x = np.linspace(params[param_IDs[j]]-3.2*np.sqrt(F_reduced[j,j]),params[param_IDs[j]]+3.2*np.sqrt(F_reduced[j,j]), 100)
                    y = np.exp(-0.5*(x-params[param_IDs[j]])/F_reduced[j,j]*(x-params[param_IDs[j]]))/np.sqrt(2*np.pi*F_reduced[j,j])
                    plt.subplot(gs[i,j])
                    if PlotLabels is not None:
                        plt.plot(x,y,clist2[k],label = PlotLabels[k]*(i+1==num_panels))
                    else:
                        plt.plot(x,y,clist2[k])
                    if (k==0) and (MCMC is not None) and (isinstance(MCMC,evil.MCMC)):
                        xdata = MCMC.data[:,:,j].ravel()+mean_subtracted*(-np.mean(MCMC.data[:,:,j])+params[param_IDs[j]])
                        plt.hist(xdata,bins=50,histtype='step',color='k',normed=True);
                    
                    xlabels = np.linspace(labels[i,j,2],labels[i,j,3],3)
                    
                    if i+1 == num_panels:
                        plt.xticks(xlabels,rotation=45)
                        plt.xlabel(Paramlabels[param_IDs[i]],fontsize=10*figscale)
                    else:
                        plt.xticks(xlabels,[])
                    plt.yticks([])
                    plt.xlim(limits[i,j,2],limits[i,j,3])
    
    if PlotLabels is not None:
        plt.legend(loc=[0.05,1.05],fontsize=24*figscale)
    
    return fig
                    
            
def reconstruct_subhalo_tesselation(subx,suby):
    '''given list of x and y coordinates, return the subhalo simplex tesselation'''
    
    Nsubs = len(subx)
    Num_levels=10
    
    # Begin the tesellation
    T1_cent = np.array([subx[0],suby[0]])
    T1_size = np.linalg.norm(np.array([subx[1],suby[1]])-T1_cent)
    tesl = np.zeros([Nsubs,3,2])
    
    for j in range(Num_levels):
        
        A = tesl[:,0,0]*tesl[:,1,1]+tesl[:,0,1]*tesl[:,2,0]+tesl[:,1,0]*tesl[:,2,1]
        A -=tesl[:,0,0]*tesl[:,2,1]+tesl[:,0,1]*tesl[:,1,0]+tesl[:,1,1]*tesl[:,2,0]
        if np.sum(A==0) ==0:
            break
        
        for k in range(Nsubs):
            Ti_cent = np.array([subx[k],suby[k]])
            Ti_size = T1_size/(2.0**j)
            
            # Try forward facing triangle
            vert1 = Ti_cent+Ti_size*np.array([np.cos(0),np.sin(0)])
            vert2 = Ti_cent+Ti_size*np.array([np.cos(2*np.pi/3.0),np.sin(2*np.pi/3.0)])
            vert3 = Ti_cent+Ti_size*np.array([np.cos(4*np.pi/3.0),np.sin(4*np.pi/3.0)])
            
            B1 = (subx-vert1[0]) * (vert1[1]-vert2[1]) / (vert1[0]-vert2[0]) + Ti_cent[1]
            B2 = vert2[0]
            B3 = (subx-vert1[0]) * (vert3[1]-vert1[1]) / (vert3[0]-vert1[0]) + Ti_cent[1]
            
            if (np.sum((suby<B1+10**-5) & (subx>B2-10**-5) & (suby>B3-10**-5))==1) & (tesl[k,0,0] ==0):
                tesl[k,0,:] = vert1
                tesl[k,1,:] = vert2
                tesl[k,2,:] = vert3
            
            # Try backward facing triangle
            vert1 = Ti_cent+Ti_size*np.array([np.cos(np.pi),np.sin(np.pi)])
            vert2 = Ti_cent+Ti_size*np.array([np.cos(2*np.pi/3.0+np.pi),np.sin(2*np.pi/3.0+np.pi)])
            vert3 = Ti_cent+Ti_size*np.array([np.cos(4*np.pi/3.0+np.pi),np.sin(4*np.pi/3.0+np.pi)])
            
            B1 = (subx-vert1[0]) * (vert1[1]-vert2[1]) / (vert1[0]-vert2[0]) + Ti_cent[1]
            B2 = vert2[0]
            B3 = (subx-vert1[0]) * (vert3[1]-vert1[1]) / (vert3[0]-vert1[0]) + Ti_cent[1]
            
            if (np.sum( (suby>B1-10**-5) & (subx<B2+10**-5) & (suby<B3+10**-5)) ==1) & (tesl[k,0,0] == 0):
                tesl[k,0,:] = vert1
                tesl[k,1,:] = vert2
                tesl[k,2,:] = vert3
                
    # Reshape the constructed tesellation and assign sequential indices for tripcolor
    tesl2 = tesl.reshape(-1,2)
    inds  = np.arange(tesl2.shape[0]).reshape(tesl.shape[0],3)
    
    
    return tesl, inds


    
def load_binary(binaryfile):
    with open(binaryfile,'rb') as file:
        filecontent = file.read()
        data = np.array(struct.unpack("d"*(len(filecontent)//8),filecontent))
    file.close()
    return data
    
def write_binary(array,binaryfile):
    with open(binaryfile,'wb') as file:
        data = struct.pack('d'*len(array),*array)
        file.write(data)
    file.close()