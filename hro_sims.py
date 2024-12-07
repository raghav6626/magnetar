'''
Taken from HROexample.py and hro.py; A reduced simple version that works with simulations

Use function hroLITE to calculate the HRO between a given 
2D density map and the B-field vectors in the plane (B_x, B_y)


Raghav Arora Dec 2024

'''
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from hro import *
# ===================================================================================================
def roangles(Imap, Qmap, Umap, ksz=1, mask=None, mode='reflect', convention='Planck', debug=False):
    # Calculates the relative orientation angle between the density structures and the magnetic field following the method
    # presented in Soler, et al. ApJ 774 (2013) 128S
    #
    # INPUTS
    # Imap - Intensity or column density map
    # Qmap - Stokes Q map
    # Upam - Stokes U map
    # ksz - pixel units...
    # OUTPUTS
    # phi - relative orientation angle between the column density and projected magnetic field.

    if (mask is None):
       mask=np.ones_like(Imap)
   
    # psi=0.5*np.arctan2(Umap,Qmap)	
    # ex=-np.sin(psi)
    # ey=np.cos(psi) 
    # angleb=np.arctan2(ey,-ex)
    bx=bx; by=by
    ex = -by; 
    ey = bx;

    '''
    Apply Gaussian filters for the derivative calculation.
    See Sec 2.1.1. in Soler et. al. 2013
    Gaussian convolved 
    x is the second index 
    y is the first index
    Be careful here. The order=[1,0] or [0,1] decides the axis of the differentiation
    '''

    sImap=ndimage.filters.gaussian_filter(Imap, [ksz, ksz], order=[0,0], mode=mode)
    dIdx =ndimage.filters.gaussian_filter(Imap, [ksz, ksz], order=[0,1], mode=mode)
    dIdy =ndimage.filters.gaussian_filter(Imap, [ksz, ksz], order=[1,0], mode=mode)
    angleg=np.arctan2(dIdx,dIdy)

    normgrad=np.sqrt(dIdx**2+dIdy**2)
    unidIdx=dIdx/np.sqrt(dIdx**2+dIdy**2)
    unidIdy=dIdy/np.sqrt(dIdx**2+dIdy**2)

    cosphi=(dIdy*ey+dIdx*ex)/normgrad
    sinphi=(dIdy*ex-dIdx*ey)/normgrad

    #phi=np.arctan2(np.abs(dIdy*ex-dIdx*ey), dIdy*ey+dIdx*ex)
    phi=np.arctan2(np.abs(sinphi), cosphi)
    #phi=np.arccos(cosphi)
    #phi=np.arcsin(sinphi)
    bad=((dIdx**2+dIdy**2)==0.).nonzero()    #np.logical_or((dIdx**2+dIdy**2)==0., (Qmap**2+Umap**2)==0.).nonzero()	
    phi[bad]=np.nan
    bad=((Qmap**2+Umap**2)==0.).nonzero()
    phi[bad]=np.nan
    if np.any(mask < 1.):
       bad=(mask < 1.).nonzero()
       phi[bad]=np.nan

    # ------------------------------------------------------------------------------------
    '''
    Just for creating the plot.
    '''
    vecpitch=10
    xx, yy, gx, gy = vectors(Imap, dIdx*mask, dIdy*mask, pitch=vecpitch) 
    xx, yy, ux, uy = vectors(Imap, bx*mask, by*mask, pitch=vecpitch)   
    if np.array_equal(np.shape(Imap), np.shape(mask)):
       bad=(mask==0.).nonzero()
       phi[bad]=np.nan    
 
    # Debugging -------------------------------------------------------------------------
    if (debug):
       levels=np.nanmean(Imap)+np.array([0.,1.,2.,3.,5.,7.])*np.nanstd(Imap)

       fig = plt.figure(figsize=(7.0,7.0))
       plt.rc('font', size=10)
       ax1=plt.subplot(111)
       im=ax1.imshow(np.abs(np.rad2deg(np.arctan(np.tan(phi)))), origin='lower', cmap='cividis')
       ax1.quiver(xx, yy, gx, gy, units='width', color='red',   pivot='middle', scale=25., headlength=0, headwidth=1, label=r'$\nabla I$')
       ax1.quiver(xx, yy, ux, uy, units='width', color='black', pivot='middle', scale=25., headlength=0, headwidth=1, label=r'$B_{\perp}$')
       ax1.contour(Imap, origin='lower', colors='black', levels=levels, linewidths=1.0)
       cbar=fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
       cbar.ax.set_title(r'$\phi$')
       plt.legend()
       plt.show()
       #import pdb; pdb.set_trace()

    return np.arctan(np.tan(phi))


# ===================================================================================================
def hroLITE(Imap, Qmap, Umap, steps=10, hsize=15, minI=None, mask=None, ksz=1, showplots=False, w=None, convention='Planck', outh=[0,4,9], savefig=False, prefix='', segmap=None, debug=False):
    # Calculates the relative orientation angle between the density structures and the magnetic field.
    # INPUTS
    # Imap - Intensity or column density map
    # Qmap - Stokes Q map
    # Umap - Stokes U map
    # mask -     

    sz=np.shape(Imap)

    if w is None:
        w=np.ones_like(Imap)
    assert w.shape == Imap.shape, "Dimensions of Imap and w must match"

    '''
    Calculation of relative orientation angles
    Equation 4 in Soler and Hennebelle 2017.
    '''
    phi=roangles(Imap, Qmap, Umap, mask=mask, ksz=ksz, convention=convention, debug=debug)

    if segmap is None:
        stepmap=Imap.copy()
    else:
        assert Imap.shape == segmap.shape, "Dimensions of Imap and segmap must match" 
        stepmap=segmap.copy()

    if (minI is None):
        print("Minimum value not specified")
        minI=np.nanmin(stepmap[(mask > 0.).nonzero()])

    if np.array_equal(np.shape(Imap), np.shape(mask)):
        bad=np.isnan(Imap).nonzero()
        mask[bad]=0.
        bad=np.isnan(stepmap).nonzero()
        mask[bad]=0.
    else:
        mask=np.ones_like(Imap)
        bad=np.isnan(Imap).nonzero()
        mask[bad]=0.
        bad=np.isnan(stepmap).nonzero()
        mask[bad]=0.

    if minI is None:
        minI=np.nanmin(stepmap)
    bad=(stepmap <= minI).nonzero()
    mask[bad]=0.
    bad=np.isnan(phi).nonzero()
    mask[bad]=0.

    good=(mask > 0.).nonzero()
    bad=(mask < 1.).nonzero()
    stepmap[bad]=np.nan


    '''
    Now we create the HRO using the angles. 

    '''

    hist, bin_edges = np.histogram(stepmap[good], bins=int(0.75*np.size(Imap)))     
    bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
    chist=np.cumsum(hist)
    pitch=np.max(chist)/float(steps)

    hsteps=pitch*np.arange(0,steps+1,1)	
    Isteps=np.zeros(steps+1)

    for i in range(0,steps):
        good=np.logical_and(chist>hsteps[i], chist<=hsteps[i+1]).nonzero()
        Isteps[i]=np.min(bin_centre[good])	
    Isteps[np.size(Isteps)-1]=np.nanmax(stepmap)

    for i in range(0,steps+1):
        diff=np.abs(i*pitch-chist)
        Isteps[i]=np.mean(bin_centre[(diff==np.nanmin(diff)).nonzero()])

    # Preparing output of the HRO
    hros=np.zeros([steps,hsize])
    s_hros=np.zeros([steps,hsize])

    Smap=np.nan*Imap
    xi=np.zeros(steps)
    s_xi=np.zeros(steps)
    Zx=np.zeros(steps)
    Zy=np.zeros(steps)
    s_Zx=np.zeros(steps)
    s_Zy=np.zeros(steps)
    meanphi=np.zeros(steps)
    s_meanphi=np.zeros(steps)
    mrl=np.zeros(steps)
    cdens=np.zeros(steps)

    Vd=np.zeros(steps)

    '''
    Following loop is for calculating the histogram for each 
    density bin.
    '''

    for i in range(0, np.size(Isteps)-1):
        #import pdb; pdb.set_trace()
        temp=stepmap.copy()
        temp[np.isnan(temp).nonzero()]=Isteps[i]
        temp[(mask > 1.).nonzero()]=Isteps[i]
        good=np.logical_and(temp > Isteps[i], temp <= Isteps[i+1]).nonzero()
        #good=np.logical_and(segmap > Isteps[i], segmap <= Isteps[i+1]).nonzero()
        
        #print(np.size(good))   

        hist, bin_edges = np.histogram((180/np.pi)*phi[good], bins=hsize, range=(-90.,90.))	
        bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])

        hros[i,:]=hist
        cdens[i]=np.mean([Isteps[i],Isteps[i+1]])
        Smap[good]=Isteps[i]

        TEMPxi, TEMPs_xi = roparameter(bin_centre, hist)
        xi[i]=TEMPxi
        s_xi[i]=TEMPs_xi

        outprojRS = projRS(2.*phi[good], w=w[good])     
        Zx[i]=outprojRS['Zx']
        Zy[i]=outprojRS['Zy']
        s_Zx[i]=outprojRS['s_Zx']
        s_Zy[i]=outprojRS['s_Zy']
        meanphi[i]=0.5*circmean(2.*phi[good], low=-np.pi, high=np.pi)  #0.5*outprojRS['meanphi']
        s_meanphi[i]=circstd(phi[good], low=0, high=np.pi/2.) #0.5*outprojRS['s_meanphi']
        mrl[i]=outprojRS['r']

        outprojRS0 = projRS(phi[good], w=w[good])
        Vd[i]=outprojRS0['Zx']
    '''
    xi is the parameter that quantifies the parallel/perpendicular tendency for 
    Isteps: The density bins for which this is calculated.
    '''
    return {'csteps': Isteps, 'xi': xi, 's_xi': s_xi, 'Zx': Zx, 's_Zx': s_Zx, 'Zy': Zy, 's_Zy': s_Zy, 'meanphi': meanphi, 's_meanphi': s_meanphi, 'Vd': Vd, 'asteps': bin_centre, 'hros': hros, 's_hros': s_hros, 'mrl': mrl, 'Smap': Smap, 'Amap': phi} 
