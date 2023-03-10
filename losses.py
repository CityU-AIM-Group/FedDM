#!/usr/env/bin python3.7

from functools import reduce
from operator import mul, add
from typing import List, Tuple, cast

import torch
import numpy as np
from torch import Tensor, einsum

from utils import simplex, one_hot


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.nd: str = kwargs["nd"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor, __) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        loss = - einsum(f"bk{self.nd},bk{self.nd}->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


def Focal_Cross_Entropy(probs, target, clean_mask):

    target[:,0,:,:][(clean_mask==2)] = 1
    target[:,1,:,:][(clean_mask==2)] = 0
    mask: Tensor = cast(Tensor, target.type(torch.float32))        
 
    log_p: Tensor = (probs + 1e-10).log()

    alpha = 0.25
    gamma = 2
    #FG
    fg_probs = probs[:, 1, :, :]
    weight = alpha * torch.pow(1 - fg_probs, gamma)
    fg_loss = - weight * mask[:, 1, :, :] * log_p[:, 1, :, :] # B, H, W
    #BG
    bg_probs = probs[:, 0, :, :]
    weight = (1 - alpha) * torch.pow(1 - bg_probs, gamma)
    bg_loss = - weight * mask[:, 0, :, :] * log_p[:, 0, :, :] # B, H, W        

    loss = fg_loss + bg_loss
    idx_mask = ((clean_mask==1) | (clean_mask==2)).type(torch.bool)        
    loss = loss[idx_mask].sum() / (idx_mask.sum() + 1e-10)

    return loss


class AbstractConstraints():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.nd: str = kwargs["nd"]
        self.C = len(self.idc)
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def penalty(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor, _) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        assert probs.shape == target.shape

        # b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        b: int
        b, _, *im_shape = probs.shape
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2

        value: Tensor = cast(Tensor, self.__fn__(probs[:, self.idc, ...]))
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        upper_z: Tensor = cast(Tensor, (value - upper_b).type(torch.float32)).flatten()
        lower_z: Tensor = cast(Tensor, (lower_b - value).type(torch.float32)).flatten()

        upper_penalty: Tensor = reduce(add, (self.penalty(e) for e in upper_z))
        lower_penalty: Tensor = reduce(add, (self.penalty(e) for e in lower_z))

        res: Tensor = upper_penalty + lower_penalty

        loss: Tensor = res.sum() / reduce(mul, im_shape)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss
