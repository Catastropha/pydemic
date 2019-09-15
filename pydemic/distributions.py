import torch
import numpy as np


def levy(
        shape: tuple,
        alpha: int,
        ) -> torch.Tensor:
    x = torch.empty(shape).uniform_(0, alpha)
    return torch.exp(-1 / (2 * x)) / torch.sqrt(2 * torch.tensor(np.pi).float() * torch.pow(x, 3))

