import matplotlib.pyplot as plt
import numpy as np
from hasasia.sim import sim_pta
from hasasia.sensitivity import GWBSensitivityCurve, Spectrum, Pulsar, DeterSensitivityCurve

phi = np.random.uniform(0, 2*np.pi,size=34)
theta = np.random.uniform(0, np.pi,size=34)
freqs = np.logspace(np.log10(5e-10),np.log10(5e-7),400)
psrs = sim_pta(timespan=11.4,cad=23,sigma=1e-7,
               phi=phi, theta=theta, Npsrs=34)
spectra = []
for p in psrs:
     sp = Spectrum(p, freqs=freqs)
     sp.NcalInv
     spectra.append(sp)

scGWB=GWBSensitivityCurve(spectra)
scDeter=DeterSensitivityCurve(spectra)

plt.loglog(freqs,scGWB.h_c,label='Stochastic')
plt.loglog(freqs,scDeter.h_c,label='Deterministic')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Strain Sensitivity, $h_c$')
plt.legend()
plt.show()
