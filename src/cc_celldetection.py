import glob
import os
import typing as tp

import numpy as np
import PIL.Image
import scipy.ndimage
import torch
import torchvision

from traininglib import datalib, modellib, segmentation, unet
from traininglib.segmentation import (
    margin_loss_fn,
    grid_for_patches, 
    paste_patch, 
    get_patch_from_grid,
    PatchedCachingDataset,
)
import traininglib.segmentation.connectedcomponents as concom
from .util import load_and_scale_image
from .cc_postprocessing import remove_small_objects



# assuming 10-250um cell sizes, this results in 10-250px
HARDCODED_GOOD_RESOLUTION = 1000   # px/mm

HARDCODED_MIN_CELLSIZE_UM = 10
HARDCODED_MIN_CELLSIZE_PX = \
    HARDCODED_MIN_CELLSIZE_UM * 1000 // HARDCODED_GOOD_RESOLUTION



# basic semantic segmentation
# instance segmentation during inference via connected components
class CC_CellsModule(unet.UNet):
    def __init__(self, **kw):
        super().__init__(output_channels=1, **kw)
        
        self.px_per_mm = HARDCODED_GOOD_RESOLUTION



# TODO: move upstream
class CC_CellsInference(torch.nn.Module):
    def __init__(self, module:CC_CellsModule, patchsize:int):
        super().__init__()
        self.module = module
        self.patchsize = patchsize
        self.slack = 32                                                         # TODO
        
        # in working size, not original image size
        self.min_object_size_px = \
            int(HARDCODED_MIN_CELLSIZE_UM * 1000 // module.px_per_mm)

        self._device_indicator = torch.nn.Parameter(torch.empty(0))

    def forward(
        self, 
        x:         torch.Tensor,
        px_per_mm: float,
        batchsize: int = 1
    ):
        x, grid, n, og_shape = self.prepare(x, px_per_mm)
        batch_outputs = []
        for i in range(0, n, batchsize):
            batch_output = self.process_batch(x, grid, i, n, batchsize)
            batch_outputs.extend(list(batch_output))
        raw_output = self.stitch_batch_outputs(batch_outputs, grid)
        output     = self.postprocess_output(raw_output)
        return output

    
    def prepare(self, x:torch.Tensor, px_per_mm:float):
        assert x.ndim == 3 and x.shape[-1] == 3 and x.dtype==torch.uint8, \
            'Input must be a single RGB uint8 image in HWC format'
        
        # training resolution vs input resolution ratio
        scale = self.module.px_per_mm / px_per_mm
        # original size
        H,W = x.shape[:2]
        # working size
        h,w = int(H*scale), int(W*scale)

        device = self._device_indicator.device
        x = x.to(device)

        # to f32 CHW
        x = x.permute(2,0,1) / 255
        x = datalib.resize_tensor2(x, [h,w], 'bilinear' )

        shape = torch.tensor([h,w])
        grid  = grid_for_patches(shape, self.patchsize, self.slack)
        n     = grid.reshape(-1,4).shape[0]
        
        return x, grid, n, (H,W)
    
    def process_batch(
        self, 
        x:    torch.Tensor, 
        grid: torch.Tensor, 
        i:    int, 
        n:    int,
        batchsize: int
    ) -> torch.Tensor:
        x_patches = []
        for j in range(i, min(i+batchsize, n)):
            gridcell = grid.reshape(-1,4)[j]
            x_patch  = get_patch_from_grid(x, grid, torch.tensor(j))
            x_patches.append(x_patch)

        x_batch = torch.stack(x_patches)
        output:torch.Tensor = self.module(x_batch)
        output = output.sigmoid().cpu()
        return output
    
    def stitch_batch_outputs(
        self,
        outputs:  tp.List[torch.Tensor], 
        grid:     torch.Tensor,
        og_shape: tp.Optional[tp.Tuple[int,int]] = None,
    ):
        h = int(grid[-1,-1,-2])
        w = int(grid[-1,-1,-1])
        full_output = torch.ones([1,1,h,w])
        for i,patch in enumerate(outputs):
            paste_patch(full_output, patch, grid, torch.tensor(i), self.slack)

        if og_shape is not None:
            full_output = datalib.resize_tensor2(full_output, og_shape, 'bilinear')
        return full_output
    
    def postprocess_output(self, output:torch.Tensor):
        instancemap = \
            concom.connected_components_patchwise(output > 0.5, patchsize=512)
        instancemap = instancemap[0,0]
        instancemap = \
            remove_small_objects(instancemap, threshold=self.min_object_size_px)
        instancemap = torch.unique(instancemap, return_inverse=True)[1]
        
        return {
            'raw':         output[0,0],
            'instancemap': instancemap,
        }


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

jitter = torchvision.transforms.ColorJitter(
    brightness = (0.7, 1.3),
    contrast   = (0.7, 1.3),
    saturation = (0.7, 1.3),
    hue        = (-0.15, 0.15)
)

def prepare_batch(
    raw_batch: RawBatch, 
    augment:   bool, 
    patchsize: int,
    device:    torch.device,
    whiteout_prob: float = 0.00,
):
    all_inputs  = []
    all_targets = []

    for i, item in enumerate(raw_batch):
        x = torch.as_tensor(item[0])
        t = torch.as_tensor(item[1])

        # augmentation: fully white or black image
        if torch.rand(1) < whiteout_prob:
            x = x * 0 # all black
            t = t * 0 # no target
            if torch.rand(1) < 0.5:
                x = x + 1 # all white

        x, t = datalib.random_crop(
            x[None], 
            t[None],
            patchsize=patchsize, 
            modes=['bilinear', 'nearest']
        )
        all_inputs.append(x[0])
        all_targets.append(t[0])

    inputs  = torch.stack(all_inputs)
    targets = torch.stack(all_targets)
    targets = (targets > 0.5).float()

    inputs, targets = datalib.random_rotate_flip(inputs, targets)
    inputs, targets = inputs.to(device), targets.to(device)
    # NOTE: jitter should be done on gpu, slow otherwise
    inputs = jitter(inputs)

    return inputs, targets




class CC_CellsDataset(PatchedCachingDataset):
    def __init__(self, splitfile:str, patchsize:int, px_per_mm:float):
        filepairs = datalib.load_file_pairs(splitfile)

        scale = HARDCODED_GOOD_RESOLUTION / px_per_mm
        super().__init__(filepairs, patchsize, scale)
        self.items = self.filepairs
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i:int):
        inputfile, targetfile = self.items[i]
        input  = datalib.load_image(inputfile)
        target = datalib.load_image(targetfile, mode='L')
        return input, target





class CC_Cells_CARROT(modellib.SaveableModule):
    def __init__(self, module:CC_CellsInference):
        super().__init__()
        self.module = module
        self.requires_grad_(False).eval()

    def process_image(
        self, 
        imagepath: str, 
        px_per_mm: float,
        batchsize: int = 1,
        outputshape: tp.Optional[tp.Tuple[int,int]] = None,
        progress_callback: tp.Optional[tp.Callable[[float],None]] = None
    ):
        assert outputshape is None or len(outputshape) == 2

        # NOTE: scaling down already here for more memory efficient image loading
        scale = self.module.module.px_per_mm / px_per_mm
        x, og_shape = load_and_scale_image(imagepath, scale)
        
        # image already scaled down, px_per_mm is now same as module's
        px_per_mm = self.module.module.px_per_mm
        x, grid, n, _ = self.module.prepare(x, px_per_mm)
        batch_outputs = []
        for i in range(0, n, batchsize):
            batch_output = self.module.process_batch(x, grid, i, n, batchsize)
            batch_outputs.extend(list(batch_output))
            if progress_callback is not None:
                progress_callback( i/n )
        if outputshape is None:
            outputshape = og_shape
        raw_output = \
            self.module.stitch_batch_outputs(batch_outputs, grid, outputshape)
        output = (raw_output > 0.5).cpu().numpy()[0,0]
        return output



