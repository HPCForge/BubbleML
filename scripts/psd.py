r"""
Estimate the radially averaged power spectral density
"""

import matplotlib.pyplot as plt
from pysteps.utils.spectral import rapsd
import torch

unet_job_id = None
fno_job_id = None

# 1. load temp predictions for sim, UNet2d, and FNO

output_tensors = {
    'sim': 'test_im/temp/{unet_job_id}/sim_output.pt'
    'unet': 'test_im/temp/{unet_job_id}/model_output.pt'
    'fno': 'test_im/temp/{fno_job_id}/model_output.pt'
}

# 2. go through every 10-th timestep, plotting rapsd
