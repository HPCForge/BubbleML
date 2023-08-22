r"""
Estimate the radially averaged power spectrum
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import stats
from scipy import signal

output_tensors = {
    'Simulation Temperature': 'scripts/data/vel_unet_mod_push/velx_label.pt',
    'UNet$_{mod}$ Temperature': 'scripts/data/vel_unet_mod/velx_output.pt',
    'P-UNet$_{mod}$ Temperature': 'scripts/data/vel_unet_mod_push/velx_output.pt',
    'UNO Temperature': 'scripts/data/uno/velx_output.pt'
}

output_tensors = {
    'Simulation': 'scripts/data/vel_unet_mod_push/sim_ouput.pt',
    'UNet$_{mod}$': 'scripts/data/vel_unet_mod/model_ouput.pt',
    'P-UNet$_{mod}$': 'scripts/data/vel_unet_mod_push/model_ouput.pt',
    'UNO': 'scripts/data/uno/model_ouput.pt'
}

data = [(name, torch.load(pth)) for (name, pth) in output_tensors.items()]

power = {}

plt.rc("font", family="serif", size=18, weight="bold")
plt.rc("axes", labelweight="bold")

steps = [0, 15, 30, 60]

fig, ax = plt.subplots(1, len(steps), figsize=(15, 5))

for idx, time in enumerate(steps):
    ax[idx].set_title(f'Step {time}')
    ax[idx].set_yscale('log')
    if idx == 0:
        ax[idx].set_ylabel('Magnitude')
    ax[idx].set_xlabel('Frequency')
    #ax[idx].set_ylim([0, 100000])
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
        N = 8
        f = np.array([1.0 / N for _ in range(N)])


        ax[idx].plot(kvals, signal.lfilter(f, 1, Abins), label=name, linewidth=2)

        #plt.plot(psd[::-1], label=name)
        #plt.plot(freq, label=name+'freq')
#plt.tight_layout()
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(f'psd_time.png')
plt.close()

