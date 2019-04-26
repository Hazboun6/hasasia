# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
from . import sensitivity as sens

__all__ = ['',
           '',
           ]

day_sec = 24*3600
yr_sec = 365.25*24*3600
c = 2.998e8
G=1

class sky_map():
    '''
    Class to make sky maps for deterministic PTA gravitational wave signals.
    Calculated in terms of $\hat{n}=-\hat{k}$.
    Note: $\hat{l}=> -\hat{l}$ and $\hat{m}=>\hat{m}$.
    '''
    def __init__(self,spec):
        self.pos = - khat(spec.theta, spec.phi)

    def eplus(theta,phi):
        l = lhat(theta,phi)
        m = mhat(theta,phi)
        return np.outer(-l, -l) - np.outer(m, m)

    def ecross(theta,phi):
        l = lhat(theta,phi)
        m = mhat(theta,phi)
        return np.outer(-l, m) + np.outer(m, -l)

    def D(theta,phi,pos):
        return 0.5 * np.outer(pos,pos) / (1 + np.dot(pos, -khat(theta,phi)))

    def sky_term(theta,ph,self.pos):
        pos = self.pos
        first = np.abs(np.einsum('ij,ij', D(theta,phi,pos), eplus(theta,phi)))
        second = np.abs(np.einsum('ij,ij', D(theta,phi,pos), ecross(theta,phi)))
        return first**2 + second**2

    def h_circ(M_c,D_L,f):
        return (4*c/D_l) * np.power(G*M_c/c**2,5/3) * np.power(np.pi*f, 2/3)

def khat(theta,phi):
    '''Returns $\hat{k}$ from paper. Also equal to $-\hat{r}=-\hat{n}$.'''
    return np.array([-np.sin(theta)*np.cos(phi),
                     -np.sin(theta)*np.sin(phi),
                     np.cos(theta)])

def lhat(theta,phi):
    '''Returns $\hat{l}$ from paper. Also equal to $-\hat{phi}$.'''
    return np.array([np.sin(phi),-np.cos(phi),0])

def mhat(theta,phi):
    '''Returns $\hat{m}$ from paper. Also equal to $-\hat{theta}$.'''
    return np.array([-np.cos(theta)*np.cos(phi),
                     -np.cos(theta)*np.sin(phi),
                     np.sin(theta)])
