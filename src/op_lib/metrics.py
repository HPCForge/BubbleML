r"""
PyTorch computes metrics w.r.t. each item in a batch individually.
It then sums or averages those individual metrics. These implementations
take the same approach.
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import math
import numpy as np
import numba as nb

from .losses import LpLoss

@dataclass
class Metrics:
    mae: float
    rmse: float
    relative_error: float
    max_error: float
    boundary_rmse: float
    interface_rmse: float
    fourier_low: float
    fourier_mid: float
    fourier_high: float

    def __str__(self):
        return f"""
            MAE: {self.mae}
            RMSE: {self.rmse}
            Relative Error: {self.relative_error}
            Max Error: {self.max_error}
            Boundary RMSE: {self.boundary_rmse}
            Interface RMSE: {self.interface_rmse}
            Fourier
                - Low: {self.fourier_low}
                - Mid: {self.fourier_mid}
                - High: {self.fourier_high}
        """

def compute_metrics(pred, label, dfun):
    low, mid, high = fourier_error(pred, label, 8, 8)
    return Metrics(
        mae=mae(pred, label),
        rmse=rmse(pred, label),
        relative_error=relative_error(pred, label),
        max_error=max_error(pred, label),
        boundary_rmse=boundary_rmse(pred, label),
        interface_rmse=interface_rmse(pred, label, dfun),
        fourier_low=low,
        fourier_mid=mid,
        fourier_high=high
    )

def write_metrics(pred, label, iter, stage, writer):
    writer.add_scalar(f'{stage}/MAE', mae(pred, label), iter)
    writer.add_scalar(f'{stage}/RMSE', rmse(pred, label), iter)
    writer.add_scalar(f'{stage}/MaxERror', max_error(pred, label), iter)

def mae(pred, label):
    return F.l1_loss(pred, label)

def relative_error(pred, label):
    assert pred.size() == label.size()
    loss = LpLoss(d=2, reductions='mean')
    return loss(pred, label)

def rmse(pred, label):
    r""" Assumes input has shape [b x h x w]
    """
    assert pred.size() == label.size()
    batch_size = pred.size(0)
    var_size = pred[0].numel() 
    sum_dim = 1 if pred.dim() == 2 else [1, 2]
    mses = ((pred - label) ** 2).sum(dim=sum_dim) / var_size
    return torch.sqrt(mses).sum() / batch_size

def max_error(pred, label):
    return ((pred - label) ** 2).max()

def _extract_boundary(tensor):
    r""" Extracts boundaries of a tensor [... x h x w]
    The output will have shape [... x 2h + 2w]
    """
    left = tensor[..., :, 0]
    right = tensor[..., :, -1]
    top = tensor[..., 0, :]
    bottom = tensor[..., -1, :]
    return torch.cat([left, right, top, bottom], dim=-1)

def boundary_rmse(pred, label):
    r""" assumes input has shape [b x h x w]
    """
    assert pred.size() == label.size()
    bpred = _extract_boundary(pred)
    blabel = _extract_boundary(label)
    print(bpred.size())
    return rmse(bpred, blabel) 

def interface_rmse(pred, label, dfun):
    assert pred.size() == label.size()
    assert pred.size() == dfun.size()
    mses = []
    for i in range(pred.size(0)):
        squared_error = (pred[i] - label[i]) ** 2
        mask = torch.tensor(get_interface_mask(dfun[i]))
        interface_mse = torch.mean(squared_error[mask])
        mses.append(interface_mse)
    return torch.sqrt(torch.tensor(mses)).sum() / pred.size(0)

#@nb.njit
def get_interface_mask(dgrid):
    [window, rows, cols] = dgrid.shape
    up, down = dgrid[:,1:,:]*dgrid[:,:-1,:], dgrid[:,1:,:]*dgrid[:,:-1,:]
    left, right = dgrid[:,:,1:]*dgrid[:,:,:-1], dgrid[:,:,1:]*dgrid[:,:,:-1]

    side_pad = torch.ones(window, rows, 1)
    top_pad = torch.ones(window, 1, cols)

    left = torch.cat((side_pad, left), dim = -1)
    right = torch.cat((right, side_pad), dim = -1)
    down = torch.cat((top_pad, down), dim = -2)
    up = torch.cat((up, top_pad), dim = -2)

    mask = ((left < 0) + (right < 0) + (up < 0) + (down < 0)) > 0
    return mask

def fourier_error(pred, target, Lx, Ly):
    r""" This function is taken and modified from PDEBench
    https://github.com/pdebench/PDEBench/blob/main/pdebench/models/metrics.py
    """
    ILOW = 4
    IHIGH = 12

    assert pred.dim() == 3
    assert pred.size() == target.size()
    pred_F = torch.fft.fftn(pred, dim=[1, 2])
    target_F = torch.fft.fftn(target, dim=[1, 2])
    idxs = target.size()
    nb = target.size(0)
    nx, ny = idxs[1:3]
    print(nx, ny)
    _err_F = torch.abs(pred_F - target_F) ** 2
    err_F = torch.zeros((nb, min(nx // 2, ny // 2)))
    for i in range(nx // 2):
        for j in range(ny // 2):
            it = math.floor(math.sqrt(i ** 2 + j ** 2))
            if it > min(nx // 2, ny // 2) - 1:
                continue
            err_F[:, it] += _err_F[:, i, j]
    _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny) * Lx * Ly
    low_err = torch.mean(_err_F[:ILOW])
    mid_err = torch.mean(_err_F[ILOW:IHIGH])
    high_err = torch.mean(_err_F[IHIGH:])
    return low_err, mid_err, high_err
