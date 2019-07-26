import matplotlib.pyplot as plt
import numpy as np
from hasasia.sim import sim_pta
from hasasia.sensitivity import GWBSensitivityCurve, Spectrum, Pulsar, DeterSensitivityCurve

phi = np.random.uniform(0, 2*np.pi,size=34)
theta = np.random.uniform(0, np.pi,size=34)
freqs = np.logspace(np.log10(5e-10),np.log10(5e-7),400)
psrs = sim_pta(timespan=15,cad=23,sigma=1e-7,
               phi=phi, theta=theta, Npsrs=34)
psrs3 = sim_pta(timespan=3,cad=23,sigma=1e-7,
               phi=phi, theta=theta, Npsrs=34)
spectra = []
spectra3 = []
for p in psrs:
     sp = Spectrum(p, freqs=freqs)
     sp.NcalInv
     spectra.append(sp)

for p in psrs3:
     sp = Spectrum(p, freqs=freqs)
     sp.NcalInv
     spectra3.append(sp)

scGWB1=GWBSensitivityCurve(spectra)
scGWB2=GWBSensitivityCurve(spectra3)

plt.loglog(freqs,scGWB1.h_c,label='15-year Baseline')
plt.loglog(freqs,scGWB2.h_c,label='3-year Baseline')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Characteristic Strain, $h_c$')
plt.legend()
plt.show()
