import glob
import os
import typing as tp

import numpy as np
import PIL.Image
import torch

from traininglib import datalib
from traininglib.segmentation import PatchedCachingDataset
from .util import resolve_px_per_mm



# assuming 10-250um cell sizes, this results in 10-250px
HARDCODED_GOOD_RESOLUTION = 1000   # px/mm

HARDCODED_MIN_CELLSIZE_UM = 10
HARDCODED_MIN_CELLSIZE_PX = \
    HARDCODED_MIN_CELLSIZE_UM * 1000 // HARDCODED_GOOD_RESOLUTION




class CellsDataset(PatchedCachingDataset):
    def __init__(
        self, 
        filepairs:        tp.List[tp.Tuple[str,str]], 
        patchsize:        int, 
        px_per_mm:        float | str | dict[str, float],  # input resolution 
        target_px_per_mm: float = HARDCODED_GOOD_RESOLUTION,
        *a, 
        **kw
    ):
        scale: float | list[float]
        if isinstance(px_per_mm, (int, float)):
            scale = target_px_per_mm / float(px_per_mm)
        else:
            inputs   = [inf for inf, _ in filepairs]
            resolved = resolve_px_per_mm(px_per_mm, inputs)
            scale    = [target_px_per_mm / resolved[inf] for inf in inputs]
        super().__init__(filepairs, patchsize, scale, *a, **kw)
        self.items = self.filepairs
    
    @classmethod
    def from_splitfile(cls, splitfile:str, *a, **kw):
        filepairs = datalib.load_file_pairs(splitfile)
        return cls(filepairs, *a, **kw)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i:int):
        inputfile, targetfile = self.items[i]
        input  = datalib.load_image(inputfile)
        target = datalib.load_image(targetfile, mode='L')
        return input, target




max_pool2d = torch.nn.functional.max_pool2d
def min_pool2d(x, kernel_size, stride=None, padding=0):
    return -max_pool2d(-x, kernel_size, stride=stride, padding=padding)

def delineate_instancemap(instancemap:torch.Tensor) -> torch.Tensor:
    '''Convert to a binary mask, making sure no instances touch each other'''
    assert instancemap.ndim == 2

    x0 = instancemap.float()[None,None]
    x1 = max_pool2d(x0, kernel_size=3, stride=1, padding=1)
    x2 = min_pool2d(x0, kernel_size=3, stride=1, padding=1)

    mask0 = (x0 != x1) & (x0 != 0)
    mask1 = (x0 != x2) & (x2 != 0)

    y = x0 * ~(mask0 | mask1)
    return (y != 0)[0,0]



