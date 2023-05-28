r"""
PyTorch computes metrics w.r.t. each item in a batch individually.
It then sums or averages those individual metrics. These implementations
take the same approach.
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Metrics:
    mae: float
    rmse: float
    max_error: float
    boundary_rmse: float

def compute_metrics(pred, label):
    return Metrics(
        mae=mae(pred, label),
        rmse=rmse(pred, label),
        max_error=max_error(pred, label),
        boundary_rmse=boundary_rmse(pred, label)
    )

def mae(pred, label):
    return F.l1_loss(pred, label)

def rmse(pred, label):
    r""" Assumes input has shape [b x h x w]
    """
    assert pred.size() == label.size()
    batch_size = pred.size(0)
    var_size = pred[0].numel() #.size(1) * pred.size(2) 
    sum_dim = 1 if pred.dim() == 2 else [1, 2]
    mses = ((pred - label) ** 2).sum(dim=sum_dim) / var_size
    print(mses.size())
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

def bubble_rmse(pred, label, dfun):
    #TODO: implement this, ugh!
    assert pred.size() == label.size()
    assert pred.size() == bubble_mask.size()
