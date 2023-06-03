import torch

def heatflux(temp, dfun, dy):
    r"""
    heat flux is temperature in the liquid phase in cells directly
    above the heater. dy is the grid spacing in the y direction. 
    """
    print(temp.size(), dfun.size(), dy)
    return torch.mean((dfun[:, -1] < 0) * (1 - temp[:, -1]) / dy)
