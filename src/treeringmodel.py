import os
import typing as tp

import numpy as np
import PIL.Image
import skimage.measure as skmeasure
import torch
import torchvision

from traininglib import datalib
from traininglib import unet
from traininglib.segmentation import (
    PatchedCachingDataset,
    margin_loss_fn,
    grid_for_patches, 
    paste_patch, 
    get_patch_from_grid,
)
from traininglib.modellib import BaseModel, SaveableModule
from traininglib.paths.pathdataset import FirstComponentPatchedPathsDataset
from traininglib.paths import pathutils as utils

from . import treerings_clustering_legacy as treeringlib
from .cc_celldetection import prepare_batch
from .util import load_and_scale_image



# assuming minimum 0.1mm tree ring width, this results in 25px, should be enough
HARDCODED_GOOD_RESOLUTION = 250   # px/mm

def resolution_to_scale(px_per_mm:float) -> float:
    return  HARDCODED_GOOD_RESOLUTION / px_per_mm




# basic semantic segmentation
# instance segmentation during inference via connected components
class TreeringsModule(unet.UNet):
    def __init__(self, **kw):
        super().__init__(output_channels=1, **kw)
        self.px_per_mm = HARDCODED_GOOD_RESOLUTION



RawBatch = tp.List[ tp.Tuple[torch.Tensor, torch.Tensor] ]

class TreeringsTrainStep(SaveableModule):
    def __init__(self, module:TreeringsModule, inputsize:int):
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

        recall = (y > 0.0)[t > 0].float().mean()
        
        logs = { 'bce': float(bce), 'mgn':float(mgn), 'rec':float(recall) }
        return loss, logs




class TreeringsDataset(PatchedCachingDataset):
    def __init__(
        self, 
        splitfile: str, 
        patchsize: int, 
        px_per_mm: float, 
        cachedir:  str = './cache/',
    ):
        filepairs = datalib.load_file_pairs(splitfile)

        scale = HARDCODED_GOOD_RESOLUTION / px_per_mm
        super().__init__(filepairs, patchsize, scale, cachedir=cachedir)
        self.items = self.filepairs

        targetfiles = [anf for _,anf in filepairs]
        self.items  = self._preprocess_treeringmaps(filepairs)
        
    def _preprocess_treeringmaps(
        self, 
        filepairs: tp.List[tp.Tuple[str,str]], 
    ):
        '''Perform sanity checks and normalize annotations'''
        new_cachefile = os.path.join(self.cachedir, 'cachefile2.csv')
        if os.path.exists(new_cachefile):
            return datalib.load_file_pairs(new_cachefile)

        targetfiles = [anf for _,anf in filepairs]
        cachedir = os.path.join(self.cachedir, 'targets')

        new_target_patches: tp.List[str] = []
        for i, anf in enumerate(targetfiles):
            basename = os.path.basename(anf)
            output = treeringlib.postprocess_treeringmapfile(anf, (100,100) )
            paths_yx:tp.List[np.ndarray]  = \
                [rp[0] for rp in output.ring_points_yx] \
                + [output.ring_points_yx[-1][1]]
            
            # TODO: should check if len(paths) == number of connected components
            # TODO: will fail if there is only a single ring boundary
            
            for j, gridcell in enumerate(self.grids[i].reshape(-1,4)):
                paths_yx_i = [
                    (p * self.scale) - gridcell[:2] for p in paths_yx
                ]
                paths_xy_i = [ p[:,::-1].copy() for p in paths_yx_i ]

                # re-rasterize to normalize annotations
                size = (gridcell[-2:] - gridcell[:2])[::-1].tolist()
                heatmap = utils.rasterize_multiple_paths_batched(
                    utils.encode_numpy_paths(paths_xy_i, 0), 
                    n_batches  = 1, 
                    size       = size, 
                    stepfactor = 2.0,
                ).float()[0]
                heatmap_file = os.path.join(cachedir, basename+f'.{j:03n}.png')
                datalib.write_image_tensor(heatmap_file, heatmap)
                new_target_patches.append(heatmap_file)

        # sanity check
        assert len(new_target_patches) == len(self)
    
        input_patches = [inf for inf,_ in self.items]
        new_items     = list( zip(input_patches, new_target_patches) )
        datalib.save_file_tuples(new_cachefile, new_items)
    
        return new_items
    
    def __getitem__(self, i:int):
        inputfile, targetfile = self.items[i]
        input  = datalib.load_image(inputfile)
        target = datalib.load_image(targetfile, mode='L')
        return input, target



#TODO: merge with cells model
class TreeringsInference(torch.nn.Module):
    def __init__(self, module:TreeringsModule, patchsize:int):
        super().__init__()
        self.module = module
        self.patchsize = patchsize
        self.slack = 32                                                         # TODO

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



#TODO: merge with cells model
class Treerings_CARROT(SaveableModule):
    def __init__(self, module:TreeringsInference):
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







if __name__ == '__main__':
    splitfile = '/mnt/d/Wood/2025-09-10_all_rings/splits/000_train.txt'
    ds = TreeringsDataset(splitfile, patchsize=512, px_per_mm=1000)
    ds[0]

    print('done')


