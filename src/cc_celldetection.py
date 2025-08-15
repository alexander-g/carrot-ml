import glob
import os
import typing as tp

import torch

from traininglib import datalib, modellib, segmentation, unet
from traininglib.segmentation import (
    margin_loss_fn,
    grid_for_patches, 
    paste_patch, 
    get_patch_from_grid,
)
import traininglib.segmentation.connectedcomponents as concom





# basic instance segmentation via connected components
class CC_CellsModule(unet.UNet):
    def __init__(self, **kw):
        super().__init__(output_channels=1, **kw)



# TODO: move upstream
class CC_CellsInference(torch.nn.Module):
    def __init__(self, module:CC_CellsModule, patchsize:int):
        super().__init__()
        self.module = module
        self.patchsize = patchsize
        self.slack = 32                                                         # TODO

        self._device_indicator = torch.nn.Parameter(torch.empty(0))

    def forward(self, x:torch.Tensor, batchsize:int = 1):
        assert x.ndim == 3 and x.shape[-1] == 3 and x.dtype==torch.uint8, \
            'Input must be a single RGB uint8 image in HWC format'
        
        device = self._device_indicator.device
        x = x.to(device)

        H,W = x.shape[:2]
        # to f32 CHW
        x = x.permute(2,0,1) / 255

        all_outputs = []

        shape = torch.tensor([H,W])
        grid  = grid_for_patches(shape, self.patchsize, self.slack)
        n     = grid.reshape(-1,4).shape[0]
        for i in range(0, n, batchsize):
            x_patches = []
            for j in range(i, min(i+batchsize, n)):
                gridcell = grid.reshape(-1,4)[j]
                x_patch  = get_patch_from_grid(x, grid, torch.tensor(j))
                x_patches.append(x_patch)

            x_batch = torch.stack(x_patches)
            output:torch.Tensor = self.module(x_batch)
            output = output.sigmoid().cpu()

            all_outputs.extend( list(output) )

        full_output = torch.ones([1,1,H,W])
        for i,patch in enumerate(all_outputs):
            paste_patch(full_output, patch, grid, torch.tensor(i), self.slack)
        
        instancemap = \
            concom.connected_components_patchwise(full_output > 0.5, patchsize=512)
        instancemap = instancemap[0,0]
        instancemap = remove_small_objects(instancemap, threshold=10)
        instancemap = torch.unique(instancemap, return_inverse=True)[1]
        
        return {
            'raw':         full_output[0,0],
            'instancemap': instancemap,
        }


def remove_small_objects(instancemap:torch.Tensor, threshold:int) -> torch.Tensor:
    assert instancemap.ndim == 2 and instancemap.dtype == torch.int64
    
    labels, counts = torch.unique(instancemap, return_counts=True)
    good_labels = labels[counts > threshold]

    mask = torch.isin(instancemap, good_labels) & (instancemap != 0)
    return mask * instancemap



RawBatch = tp.List[ tp.Tuple[torch.Tensor, torch.Tensor] ]

class CC_CellsTrainStep(modellib.SaveableModule):
    def __init__(self, module:CC_CellsModule, inputsize:int):
        super().__init__()
        self.module    = module
        self.inputsize = inputsize
        self._device_indicator = torch.nn.Parameter(torch.empty(0))

    def forward(self, raw_batch:RawBatch):
        x,t = prepare_batch(
            raw_batch,
            augment   = True,
            patchsize = self.inputsize,
            device    = self._device_indicator.device
        )
        y = self.module(x)

        bce_fn = torch.nn.functional.binary_cross_entropy_with_logits
        bce    = bce_fn(y, t)
        mgn    = margin_loss_fn(y, t.bool())
        loss   = bce + mgn * 0.1
        
        logs = { 'bce': float(bce), 'mgn':float(mgn) }
        return loss, logs



def prepare_batch(
    raw_batch: RawBatch, 
    augment:   bool, 
    patchsize: int,
    device:    torch.device,
):
    inputs  = torch.stack([
        torch.as_tensor(item[0]) for item in raw_batch
    ])
    targets = torch.stack([
        torch.as_tensor(item[1]) for item in raw_batch
    ])
    targets = (targets > 0.5).float()

    inputs, targets = datalib.random_crop(
        inputs, 
        targets, 
        patchsize=patchsize, 
        modes=['bilinear', 'nearest']
    )
    inputs, targets = datalib.random_rotate_flip(inputs, targets)
    inputs, targets = inputs.to(device), targets.to(device)

    return inputs, targets




class CC_CellsDataset:
    def __init__(self, splitfile:str):
        self.items = datalib.load_file_pairs(splitfile)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i:int):
        inputfile, targetfile = self.items[i]
        input  = datalib.load_image(inputfile)
        target = datalib.load_image(targetfile, mode='L')
        return input, target


