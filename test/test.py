import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift

from src.CSVD import CSVD

t=np.arange(0,1024)*.01
ampl = np.random.normal(1,0.2,(1000,1))
fr = np.random.normal(-15,0.1,(1000,1))
sig1 = ampl * np.exp(-2*t) *np.exp(2*np.pi*fr*t*1j)

ampl2 = np.random.normal(1,0.2,(1000,1))
fr2 = np.random.normal(0,0.1,(1000,1))
sig2 = ampl2 * np.exp(-2*t) *np.exp(2*np.pi*fr2*t*1j)

sig = sig1 + sig2
noise = np.random.normal(0,1,(sig.shape)) + 1j*np.random.normal(0,1,(sig.shape))
sig = sig + 0.1*noise

csvd = CSVD(sig.T, 0.01)

sig_ = csvd.remove('auto',[-20,-10],3)
plt.plot(fftshift(fft(sig[0,:])).T)
plt.plot(fftshift(fft(sig_[:,0])).T)
plt.legend(['Orginal signal', 'Water-removed signal'])
plt.savefig('example.jpg')
plt.show()

