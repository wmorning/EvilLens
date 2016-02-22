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
    for i in range(Numpars):
        plt.subplot(gs[i//Ncols,i % Ncols])
        for j in range(MCMC.Nwalkers):
            plt.plot(MCMC.data[j,Tburn:Niter,i])
        if paramslist is not None:
            plt.ylabel(paramslist[i])
        else:
            pass
        if i+Ncols >= Numpars:
            plt.xlabel('Number of iterations')
            plt.xticks(np.linspace(Tburn,Niter,5))
        else:
            plt.xticks(np.linspace(Tburn,Niter,5),[])
        
        plt.xlim(Tburn,Niter)
    
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
            
            
    return
    
def Plot_chi2(MCMC,Tburn=0,Title=None,Filename=None,figsize=[5,5]):
    
    fig = plt.figure(figsize=figsize)
    
    if Tburn > MCMC.Niter:
        print('Burn Time is larger than number of iterations.  Setting to zero')
        Tburn = 0
    else:
        pass
    
    for i in range(MCMC.Nwalkers):
        plt.plot(MCMC.chi2[i,Tburn:])
    
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
            
    return

    
    
def Plot_dirty_image(Vis_data,Vis_model,u,v,Img_L = 4.0, Img_cent=[0,0]  \
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
    img_mo = np.zeros([Num_pixels,Num_pixels])
        
       
    for i in range(Num_pixels):
        for j in range(Num_pixels):
            img_da[i,j] = 2*np.sum((Vis_data *np.exp(2j*np.pi*(u*x[j]+v*y[i]))).real)
            img_mo[i,j] = 2*np.sum((Vis_model*np.exp(2j*np.pi*(u*x[j]+v*y[i]))).real) 
        
    resid = (img_da-img_mo)/np.std(img_da-img_mo)
        
    # convert x,y back to arcsec
    x*=(3600*180/np.pi)
    y*=(3600*180/np.pi)
        
    fig = plt.figure(figsize=figsize)
    if Title is not None:
        plt.suptitle(Title,fontsize=22)
    gs = gridspec.GridSpec(1,3)
    gs.update(left=0.05,right=0.9,wspace=0.03)
        
    plt.subplot(gs[0,0])
    if Flipped is False:
        plt.imshow(img_da,origin='lower',extent=(np.min(x),np.max(x) \
                ,np.min(y),np.max(y)), cmap='cubehelix'         \
                ,interpolation='nearest', vmin= np.min(img_da)   \
                ,vmax=np.max(img_da))
    else:
        plt.imshow(img_da,origin='lower',extent=(np.min(x),np.max(x) \
                ,np.max(y),np.min(y)), cmap='cubehelix'         \
                ,interpolation='nearest', vmin= np.min(img_da)   \
                ,vmax=np.max(img_da))
        plt.ylim(np.min(y),np.max(y))
    plt.title('Data')
    plt.ylabel('y (arcseconds)',fontsize=14)
    plt.xlabel('x (arcseconds)',fontsize=14)
        
    plt.subplot(gs[0,1])
    if Flipped is False:
        plt.imshow(img_mo,origin='lower',extent=(np.min(x),np.max(x) \
                ,np.min(y),np.max(y)), cmap='cubehelix'          \
                ,interpolation='nearest', vmin= np.min(img_da)   \
                ,vmax=np.max(img_da))
    else:
        plt.imshow(img_mo,origin='lower',extent=(np.min(x),np.max(x) \
                ,np.max(y),np.min(y)), cmap='cubehelix'          \
                ,interpolation='nearest', vmin= np.min(img_da)   \
                ,vmax=np.max(img_da))
        plt.ylim(np.min(y),np.max(y))
    plt.yticks([])
    plt.title('Model')
    plt.xlabel('x (arcseconds)',fontsize=14)
        
    plt.subplot(gs[0,2])
    if Flipped is False:
        plt.imshow(resid,origin='lower',extent=(np.min(x),np.max(x) \
                ,np.min(y),np.max(y)), cmap='cubehelix'          \
                ,interpolation='nearest')
    else:
        plt.imshow(resid,origin='lower',extent=(np.min(x),np.max(x) \
                ,np.max(y),np.min(y)), cmap='cubehelix'          \
                ,interpolation='nearest')
        plt.ylim(np.min(y),np.max(y))
    plt.yticks([])
    plt.title('Residuals')
    plt.xlabel('x (arcseconds)',fontsize=14)
        
    cax = fig.add_axes([0.92,0.2,0.02,0.62])
    plt.colorbar(cax=cax,label=r'$\sigma$')
        
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
                
    return
        
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
        else:
            plt.plot(MCMC.GR2[:,i]-1)
        plt.ylim(yrange[0],yrange[1])
        plt.yscale('log')
        plt.plot([0,len(MCMC.GR[:,i])],[0.01,0.01],'k--')
        plt.plot([0,len(MCMC.GR[:,i])],[0.1,0.1],'r--')
        plt.plot([0,len(MCMC.GR[:,i])],[MCMC.alpha,MCMC.alpha],'g--')
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
                
    return
    
    
    
def Plot_Triangle(MCMC,paramslist=None,Title=None,Filename=None,levels=[0.68,0.95],truths=None):
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
    
    
    data = MCMC.data.reshape(-1,MCMC.Nparameters)
    fig = corner.corner(data,levels=levels,labels=paramslist,truths=truths)
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
                
    return
    
def Plot_source(sourcedir,Npixels,source_L,sourcecent,includecaustics=True,causticX=None,\
                causticY=None,Ngridlines=None,figsize=[10,10],Title=None,Filename=None,Flipped=False):
    src = np.loadtxt(sourcedir).reshape([Npixels,Npixels]).T
    
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1,2)
    gs.update(left=0.1,right=0.9,top =1.35,wspace=0.03)
    plt.subplot(gs[0,0])
    if Flipped is False:
        plt.imshow(src,origin='lower',interpolation='nearest',cmap='cubehelix',   \
                extent=(sourcecent[0]-source_L/2.0,sourcecent[0]+source_L/2.0,\
                        sourcecent[1]-source_L/2.0,sourcecent[1]+source_L/2.0))
    else:
        plt.imshow(src,origin='lower',interpolation='nearest',cmap='cubehelix',   \
                extent=(sourcecent[0]-source_L/2.0,sourcecent[0]+source_L/2.0,\
                        -sourcecent[1]+source_L/2.0,-sourcecent[1]-source_L/2.0))
    plt.ylabel('y (arcsec)')
    plt.xlabel('x (arcsec)')
    if Title is not None:
        plt.suptitle(Title)
    
    if (includecaustics is True) and (causticX is not None) and (causticY is not None):
        caustX = np.loadtxt(causticX)
        caustY = np.loadtxt(causticY)
        
        caustX *= 3600*180/np.pi
        caustY *= 3600*180/np.pi
        
        if Ngridlines is not None:
            step = caustX.shape[0]//Ngridlines
            Nlines = Ngridlines
        else: 
            step = 1
            Nlines = caustX.shape[0]
        if Flipped is False:
            for i in range(Nlines):
                plt.plot(caustX[step*i,:],caustY[step*i,:],'w-',linewidth=0.2)
                plt.plot(caustX[:,step*i],caustY[:,step*i],'w-',linewidth=0.2)
        else:
            for i in range(Nlines):
                plt.plot(caustX[step*i,:],-caustY[step*i,:],'w-',linewidth=0.2)
                plt.plot(caustX[:,step*i],-caustY[:,step*i],'w-',linewidth=0.2)

            
    
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
        plt.imshow(src,origin='lower',interpolation='nearest',cmap='cubehelix',   \
                extent=(sourcecent[0]-source_L/2.0,sourcecent[0]+source_L/2.0,\
                        sourcecent[1]-source_L/2.0,sourcecent[1]+source_L/2.0))
                        
        plt.yticks(np.linspace(-source_L/2.0,source_L/2.0,5)+sourcecent[1],[])
        plt.ylim(-source_L/2.0+sourcecent[1],source_L/2.0+sourcecent[1])
        plt.xlim(-source_L/2.0+sourcecent[0],source_L/2.0+sourcecent[0])
    else:
        plt.imshow(src,origin='lower',interpolation='nearest',cmap='cubehelix',   \
                extent=(sourcecent[0]-source_L/2.0,sourcecent[0]+source_L/2.0,\
                        -sourcecent[1]+source_L/2.0,-sourcecent[1]-source_L/2.0))
        plt.yticks(np.linspace(-source_L/2.0,source_L/2.0,5)+sourcecent[1],[])
        plt.ylim(-source_L/2.0-sourcecent[1],source_L/2.0-sourcecent[1])
        plt.xlim(-source_L/2.0+sourcecent[0],source_L/2.0+sourcecent[0]) 
    plt.xlabel('x (arcsec)')
    
    
    
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
                
    return
    
    
def Compare_Triangles(MCMC1,MCMC2,paramslist=None,Title=None,Filename=None,levels=[0.68,0.95],truths=None):
    
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
        
    fig = corner.corner(d1,levels = levels, labels=paramslist , truths=truths)
    corner.corner(d2,fig=fig,levels=levels,color='blue');
    
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
                
    return
    
    
def Compare_chains(MCMC1,MCMC2,paramslist=None,Numpars=None,Tburn=None,Niter=None,figsize=[5,5], \
                Nrows=4, Ncols=4,PlotTitle=None, Filename=None):
                    
    if paramslist is not None:
        paramslist = paramslist
    
    if Numpars is not None:
        Numpars = Numpars
    else:
        Numpars = MCMC1.Nparameters
    
    if Tburn is not None:
        Tburn = Tburn
    else:
        Tburn = 0
        
    if Niter is not None:
        Niter1 = Niter
        Niter2 = Niter
    else:
        Niter1 = MCMC1.Niter
        Niter2 = MCMC2.Niter
        
    assert Numpars <= Nrows*Ncols
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(Nrows,Ncols)
    gs.update(left=0.05,right=0.95,wspace=0.5,hspace=0.1)
    for i in range(Numpars):
        plt.subplot(gs[i//Ncols,i % Ncols])
        for j in range(MCMC1.Nwalkers):
            plt.plot(MCMC1.data[j,Tburn:Niter1,i],'k-')
        for j in range(MCMC2.Nwalkers):
            plt.plot(MCMC2.data[j,Tburn:Niter2,i],'r-')
            
        if paramslist is not None:
            plt.ylabel(paramslist[i])
        else:
            pass
        if i+Ncols >= Numpars:
            plt.xlabel('Number of iterations')
            plt.xticks(np.linspace(Tburn,np.max([Niter1,Niter2]),5))
        else:
            plt.xticks(np.linspace(Tburn,np.max([Niter1,Niter2]),5),[])
        
        plt.xlim(Tburn,np.max([Niter1,Niter2]))
    
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
            
            
    return