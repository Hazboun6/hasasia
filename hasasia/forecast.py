# -*- coding: utf-8 -*-
import copy
from hasasia import sensitivity as hsen

yr_sec = 365.25*24*3600

def get_sliced_spectra(psrs, A_gwb, gamma_gwb,
                       freqs,
                       start_mjd=0.0,
                       end_mjd = 1_000_000,
                       min_tspan_cut = 3,
                       verbose=True,
                       ):
    """
    Parameters 
    ----------
    
    psrs : list of enterprise.Pulsar or list of hasasia.Pulsar objects
        List of enterprise/hasasia Pulsar objects

    A_gwb : gravitational wave background spectral amplitude

    gamma_gwb : gravitational wave background spectral index

    freqs : array
        Frequency array for the PTA in Hz

    start_mjd : float
        Start time in MJD for the slice

    end_mjd : float
        End time in MJD for the slice
    
    min_tspan_cut : float
        Minimum time span in years for the pulsar to be included in the PTA spectrum
    
    verbose : bool
        Print out the number of pulsars and the time span of the PTA spectrum

    Return
    ------
    spectra : list of hasasia.Spectrum objects
    """
    psrs_post_cut = [] # list of pulsars after timespan cuts
    psrs_copy = copy.deepcopy(psrs)
    for _ , psr in enumerate(psrs_copy):
        # filter the data around to the appropriate slice
        psr.filter_data(start_time=start_mjd, end_time=end_mjd)
        # If there are no TOAs remaining, or the time span of the pulsar is <3 cut.
        if (psr.toas.size == 0):
            pass # don't include pulsars without TOAs
        elif min_tspan_cut is None:
            psrs_post_cut.append(psr)
        elif min_tspan_cut is not None and  (hsen.get_Tspan([psr]) < min_tspan_cut*yr_sec):
            pass # don't include pulsars with less than min_tspan_cut years of data
        else:
            psrs_post_cut.append(psr)
    spectra = []
    for _ , p in enumerate(psrs_post_cut):
        sp = hsen.Spectrum(p,freqs=freqs, amp_gw=A_gwb, gamma_gw=gamma_gwb)
        sp.name = p.name
        _ = sp.NcalInv
        spectra.append(sp)
    if verbose is True:
        print(f"PTA spectrum with {len(spectra)} psrs and Tspan {round(hsen.get_Tspan([psrs_post_cut])/yr_sec, 2)} created.")
    return spectra