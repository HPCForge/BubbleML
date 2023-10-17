from torch.optim.lr_scheduler import LambdaLR

class LinearWarmupLR(LambdaLR):
    def __init__(self, optimizer, warmup_iters):
        self.warmup_iters = warmup_iters
        warmup_func = lambda current_step: min(1, current_step / self.warmup_iters)
        super().__init__(optimizer, lr_lambda=warmup_func)

