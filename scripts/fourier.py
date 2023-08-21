r"""
Estimate the radially averaged power spectrum
"""

import matplotlib.pyplot as plt
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



tensor = torch.load(output_tensors['Simulation'])

for time in range(50, 70):
    t = tensor[time].numpy() 

    f = np.fft.fft2(t)
    fshift = np.fft.fftshift(f)
    original = np.copy(fshift)

    # zero out the low frequencies
    [rows, cols] = original.shape
    crow, ccol = rows//2, cols//2
    d = 64
    fshift[crow-d:crow+d, ccol-d:ccol+d] = 0

    # subtract high frequencies from original
    # and convert back to real space
    f_ishift= np.fft.ifftshift(original - fshift)
    t_low = np.fft.ifft2(f_ishift)
    t_low = np.abs(t_low)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.flipud(t), vmin=0, vmax=1)
    ax[1].imshow(np.flipud(t_low), vmin=0, vmax=1)
    plt.savefig(f'low_freq_{time}', dpi=400)
    plt.close()

