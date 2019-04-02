# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np


def create_design_matrix(toas, RADEC=True, PROPER=False, PX=False):
    """
    Return designmatrix for quadratic spindown model + optional
    astrometric parameters

    :param toas: toa measurements [s]
    :param RADEC: (optional) Includes RA/DEC fitting
    :param PROPER: (optional) Includes proper motion fitting
    :param PX: (optional) Includes parallax fitting

    :return: M design matrix for QSD + optional astronometry

   """
    model = ['QSD', 'QSD', 'QSD']
    if RADEC:
        model.append('RA')
        model.append('DEC')
    if PROPER:
        model.append('PRA')
        model.append('PDEC')
    if PX:
        model.append('PX')

    ndim = len(model)
    designmatrix = np.zeros((len(toas), ndim))

    for ii in range(ndim):
        if model[ii] == 'QSD': #quadratic spin down fit
            designmatrix[:,ii] = toas**(ii) #Cute
        if model[ii] == 'RA':
            designmatrix[:,ii] = np.sin(2*np.pi/3.16e7*toas)
        if model[ii] == 'DEC':
            designmatrix[:,ii] = np.cos(2*np.pi/3.16e7*toas)
        if model[ii] == 'PRA':
            designmatrix[:,ii] = toas*np.sin(2*np.pi/3.16e7*toas)
        if model[ii] == 'PDEC':
            designmatrix[:,ii] = toas*np.cos(2*np.pi/3.16e7*toas)
        if model[ii] == 'PX':
            designmatrix[:,ii] = np.cos(4*np.pi/3.16e7*toas)

    return designmatrix


#########

def radec2HellingsDowns(ra, dec, displayFig=False):

    '''
    calculate hellings-downs factors for a set of M=2 or more pulsars
    specified by a vector of (ra,dec) coordinates.
    the pulsars are assumed to be distinct so that the hellings-downs
    factor only contains the earth-earth correlation term.

    ra       - array of right ascension (0 to 24 hrs)
    dec      - array of declination (+pi/2 to -pi/2)
    alphaIJ  - MxM matrix of hellings-downs factors
    IJ       - mapping from 1,2, ... Np into IJ element of MxM matrix
    thetaIJ  - angles (radians) between pairs of pulsars indexed from 1,2, ... Np
    '''

    # extract number of pulsars and error checking
    M = len(ra)

    # number of distinct pulsar pairs
    Np = np.int(M*(M-1)/2);

    # convert (ra,dec) to (theta,phi)
    theta = np.pi/2. - dec;
    phi = ra*np.pi/12.;

    cosTheta = np.cos(theta);
    sinTheta = np.sin(theta);

    # calculate hellings-downs factor for these angles
    cosThetaIJ = np.zeros((M,M))
    alphaIJ = np.zeros((M,M))
    for ii in range(M):
        for jj in range(M):
            cosThetaIJ[ii,jj] = cosTheta[ii] * cosTheta[jj] \
                                + sinTheta[ii] * sinTheta[jj] \
                                * np.cos(phi[ii] - phi[jj])
            x = (1.-cosThetaIJ[ii,jj])/2.

            if jj==ii:
                alphaIJ[ii,jj] = 1.
            else:
                alphaIJ[ii,jj] = 1.5*x*np.log(x) - 0.25*x/4. + 0.5

    # construct array of angles
    thetaIJ = np.zeros(Np)
    alphaHD = np.zeros(Np)
    IJ = np.zeros((Np,2), dtype=int)

    ind = 0
    for ii in range(M-1):
        for jj in np.arange(ii+1,M):
            IJ[ind,0] = ii
            IJ[ind,1] = jj
            thetaIJ[ind] = np.arccos(cosThetaIJ[ii,jj])
            alphaHD[ind] = alphaIJ[ii,jj]
            ind = ind+1

    # calculate rss (root-sum-squared) of hellings-downs factor
    alphaRSS = np.sqrt(np.sum(alphaHD**2));

    # plot if desired
    if displayFig==True:
        plt.figure()
        plt.plot(thetaIJ*180./np.pi, alphaHD, 'k*')
        plt.xlabel('Angle between pair of pulsars (degrees)' )
        plt.xlim([0, 180])

    return alphaIJ, alphaRSS, IJ, thetaIJ
