---
title: 'Hasasia: A Python package for Pulsar Timing Array Sensitivity Curves'
tags:
  - Python
  - astronomy
  - gravitational waves
  - pulsar timing arrays
authors:
  - name: Jeffrey S. Hazboun
    orcid: 0000-0003-2742-3321
    affiliation: 1
  - name: Joseph D. Romano
    orcid: 0000-0003-4915-3246
    affiliation: 2
  - name: Tristan L. Smith
    orcid: 0000-0003-2685-5405
    affiliation: 3
affiliations:
 - name: University of Washington Bothell
   index: 1
 - name: Texas Tech University
   index: 2
 - name: Swarthmore College
   index: 3
date: 16 September 2019
bibliography: paper.bib
---

# Summary

Gravitational waves are quickly changing the way that we view the wider
universe, enabling observations of compact objects in highly relativistic
scenarios. Gravitational-wave detectors measure the minuscule, time-dependent
perturbations to the spacetime metric. These detectors have long been
characterized by a sensitivity curve, a plot in the frequency domain, which
summarizes their ability to *detect* a given signal. Pulsar timing arrays
(PTAs) are collections of highly precise millisecond pulsars regularly
monitored for shifts in the spin period of pulsars indicative of gravitational
waves in the nanohertz regime. See @hobbs and @burke-spolaor for a review of
pulsar timing arrays and the astrophysics of nanohertz gravitational waves. The
sensitivity curves for PTAs are often overly simplified in the literature,
lacking detailed information about the fit to a pulsar's timing parameters and
assuming identical pulsar noise characteristics.

``Hasasia`` is a Python package for calculating and building accurate PTA
sensitivity curves, largely based on the formalism presented in [@hazboun:2019].
This software is designed for use by astronomers interested in sources of
nanohertz gravitational waves and PTA data analysts alike.
It uses standard Python packages, such as ``Numpy`` [@numpy] and ``Astropy``
[@astropy] to build sensitivity curves from generic PTAs of individually
constructed pulsars. ``Hasasia`` includes the ability to add time-correlated
(red) noise into the noise power spectral density of individual pulsars. The
strongest expected signal in the PTA band is the stochastic gravitational
wave background from supermassive binary black holes, which is also modeled as a red noise process. Therefore, it is important to take low-frequency noise into account when assessing the sensitivity of a PTA.

The API is designed with a general astrophysics audience in mind. In fact a number of "standard" PTA configurations are included as part of the package. It has already been made a requirement of another Python package [@gwent]. The various sensitivity curve objects in ``hasasia`` allow the
calculation of signal-to-noise ratios for a generic user-defined
gravitational-wave signal. Though the user interface is designed with the
non-expert in mind, a PTA data analyst can use real pulsar timing data to assess
the sensitivity of a given PTA.
<!--- The source code for ``Hasasia`` has been archived to Zenodo with the linked DOI: [@zenodo] --->

# Acknowledgements

JSH and JDR acknowledge subawards from the University of Wisconsin-Milwaukee for the NSF NANOGrav Physics Frontier Center (NSF PFC-1430284). JDR also
acknowledges support from start-up funds from Texas
Tech University. TLS acknowledges support from NASA
80NSSC18K0728 and the Hungerford Fund at Swarthmore College. Finally, we thank Robert Caldwell, Rutger van Haasteren and Xavi Siemens for useful discussions
and Justin Ellis for sharing some preliminary code.

# References
