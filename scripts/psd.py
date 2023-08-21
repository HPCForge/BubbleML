r"""
Estimate the radially averaged power spectrum
"""

import matplotlib.pyplot as plt
from pysteps.utils.spectral import rapsd
import torch
import numpy as np
from scipy import stats
from scipy import signal

# Note: this is data for Temp-only PB_Gravity!
output_tensors = {
    'Simulation': 'scripts/data/sim_output.pt',
    'UNet$_{bench}$': 'scripts/data/unet2d_output.pt',
    'UNO': 'scripts/data/uno_output.pt',
    'FNO': 'scripts/data/fno_output.pt'
}

data = [(name, torch.load(pth)) for (name, pth) in output_tensors.items()]

power = {}

for time in range(0, 151, 2):
    for name, tensor in data:
        print(tensor.size())
        timestep = tensor[time].numpy()
        
        fourier = np.fft.fftn(timestep)
        fourier_amp = np.abs(fourier) ** 2
        
        npix = timestep.shape[0]
        kfreq = np.fft.fftfreq(npix) * npix
        kfreq2d = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2d[0] ** 2 + kfreq[1] ** 2)
        
        knrm = knrm.flatten()
        fourier_amp = fourier_amp.flatten()

        kbins = np.arange(0.5, npix//2 + 1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])

        Abins, _, _ = stats.binned_statistic(knrm, fourier_amp,
                                             statistic='mean',
                                             bins=kbins)
        N = 5
        f = np.array([1.0 / N for _ in range(N)])
        plt.plot(kvals, signal.lfilter(f, 1, Abins), label=name, linewidth=2)
        plt.yscale('log')
        plt.ylabel('Magnitude')
        plt.xlabel('Frequency')

        #plt.plot(psd[::-1], label=name)
        #plt.plot(freq, label=name+'freq')
    plt.legend()
    plt.savefig(f'psd_time{time}')
    plt.close()

