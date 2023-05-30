r"""
PyTorch computes metrics w.r.t. each item in a batch individually.
It then sums or averages those individual metrics. These implementations
take the same approach.
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass
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

    def __str__(self):
        return f"""
            MAE: {self.mae}
            RMSE: {self.rmse}
            Relative Error: {self.relative_error}
            Max Error: {self.max_error}
            Boundary RMSE: {self.boundary_rmse}
            Interface RMSE: {self.interface_rmse}
        """

def compute_metrics(pred, label, dfun):
    return Metrics(
        mae=mae(pred, label),
        rmse=rmse(pred, label),
        relative_error=relative_error(pred, label),
        max_error=max_error(pred, label),
        boundary_rmse=boundary_rmse(pred, label),
        interface_rmse=interface_rmse(pred, label, dfun)
    )

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
        mask = torch.tensor(get_interface_mask(dfun[i].numpy()))
        interface_mse = torch.mean(squared_error[mask])
        mses.append(interface_mse)
    return torch.sqrt(torch.tensor(mses)).sum() / pred.size(0)

@nb.njit
def get_interface_mask(dgrid):
    interface = np.zeros(dgrid.shape).astype(np.bool_)
    [rows, cols] = dgrid.shape
    for i in range(rows):
        for j in range(cols):
            adj = ((i < rows - 1 and dgrid[i][j] * dgrid[i+1, j  ] <= 0) or
                   (i > 0 and dgrid[i][j] * dgrid[i-1, j  ] <= 0) or
                   (j < cols - 1 and dgrid[i][j] * dgrid[i,   j+1] <= 0) or
                   (j > 0 and dgrid[i][j] * dgrid[i,   j-1] <= 0))
            interface[i][j] = adj
    return interface
