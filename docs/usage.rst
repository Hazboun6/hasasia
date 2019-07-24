=====
Usage
=====

To use hasasia in a project::

    import hasasia.sensitivity as sens
    import hasasia.sim as sim

    phi = np.random.uniform(0, 2*np.pi,size=34)
    theta = np.random.uniform(0, np.pi,size=34)
    freqs = np.logspace(np.log10(5e-10),np.log10(5e-7),400)

    #Make a set of psrs with the same parameters
    psrs = sim_pta(timespan=11.4,cad=23,sigma=1e-7,
                   phi=phi, theta=theta, Npsrs=34)

Next make a spectra object for each pulsar. Here we calculate the
inverse-noise-weighted transmission function along the way.::

    spectra = []
    for p in psrs:
         sp = Spectrum(p, freqs=freqs)
         sp.Tf
         spectra.append(sp)

Enter the list of spectra into the GWB and deterministic sensitivity curve
classes.::

    scGWB=GWBSensitivityCurve(spectra)
    scDeter=DeterSensitivityCurve(spectra)

.. plot:: pyplot/usage_1.py

   Comparison of a sensitivity curve for a deterministic and stochastic gravitational wave signal. 

Compare this to a set of sensitivity curves made with 3-year pulsar baselines.

.. plot:: pyplot/usage_2.py
