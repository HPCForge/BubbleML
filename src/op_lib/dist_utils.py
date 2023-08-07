import os
import torch.distributed as dist

def initialize(backend):
    dist.init_process_group(backend=backend)

def dist_is_used():
    return (dist.is_available() and dist.is_initialized())

def local_rank():
    if not dist.is_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])

def rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def leader_rank():
    return 0

def is_leader_process():
    return rank() == leader_rank()
