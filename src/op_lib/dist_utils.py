import os
import torch
import torch.distributed as dist

def initialize(backend):
    dist.init_process_group(backend=backend)
    #torch.multiprocessing.set_start_method('spawn')

def dist_is_used():
    return (torch.distributed.is_available() and 
            torch.distributed.is_initialized())

def local_rank():
    assert dist_is_used()
    return int(os.environ['LOCAL_RANK'])

def rank():
    assert dist_is_used()
    return dist.get_rank()

def world_size():
    assert dist_is_used()
    return dist.get_world_size()

def leader_rank():
    assert dist_is_used()
    return 0

def is_leader_process():
    assert dist_is_used()
    return rank() == leader_rank()

def all_sum_loss(local_loss):
    local_loss_tensor = torch.tensor([local_loss]).cuda()
    dist.all_reduce(local_loss_tensor, op=dist.ReduceOp.SUM)
    return local_loss_tensor[0].item()
