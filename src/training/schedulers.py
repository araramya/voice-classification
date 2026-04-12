"""Learning rate schedulers with warmup."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Cosine annealing with linear warmup.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: number of warmup epochs (linear ramp)
        total_epochs: total training epochs
        min_lr_ratio: minimum LR as fraction of initial LR
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def get_step_scheduler(
    optimizer: Optimizer,
    step_size: int = 20,
    gamma: float = 0.5,
) -> LambdaLR:
    """Step decay scheduler."""
    def lr_lambda(epoch):
        return gamma ** (epoch // step_size)

    return LambdaLR(optimizer, lr_lambda)


def get_scheduler(optimizer: Optimizer, config) -> LambdaLR:
    """Factory function to create scheduler from config."""
    if config.scheduler == "cosine":
        return get_cosine_warmup_scheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.epochs,
        )
    elif config.scheduler == "step":
        return get_step_scheduler(optimizer)
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
