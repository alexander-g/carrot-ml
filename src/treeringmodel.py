import os
import typing as tp

import numpy as np
import PIL.Image
import skimage.measure as skmeasure
import skimage.morphology as skmorph
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
from traininglib import trainingloop

from . import treerings_clustering_legacy as treeringlib
from .util import load_and_scale_image, resolve_px_per_mm



# assuming minimum 0.1mm tree ring width, this results in 25px, should be enough
HARDCODED_GOOD_RESOLUTION = 250   # px/mm




RawBatch = tp.List[ tp.Tuple[torch.Tensor, torch.Tensor] ]



class TreeringsDataset(PatchedCachingDataset):
    def __init__(
        self, 
        filepairs:        tp.List[tp.Tuple[str,str]], 
        patchsize:        int, 
        px_per_mm:        float | str | dict[str, float], 
        target_px_per_mm: float = HARDCODED_GOOD_RESOLUTION,
        dilation:         int   = 0,
        cachedir:         str   = './cache/',
    ):
        raw_patchsize: int | list[int]
        if isinstance(px_per_mm, (int, float)):
            scale         = target_px_per_mm / float(px_per_mm)
            raw_patchsize = int(np.ceil(patchsize / scale))
        else:
            inputs        = [inf for inf, _ in filepairs]
            resolved      = resolve_px_per_mm(px_per_mm, inputs)
            raw_patchsize = [
                int(np.ceil(patchsize / (target_px_per_mm / resolved[inf])))
                for inf in inputs
            ]
        # NOTE: scale=1 because the interpolation mode='nearest' is suboptimal
        super().__init__(filepairs, raw_patchsize, scale=1.0, cachedir=cachedir)
        
        # resizing to patchsize afterwards here:
        self.items = self._preprocess_treeringmaps(self.filepairs, patchsize, dilation)
        self._resize_inputs_to_inputsize(patchsize)
    
    @classmethod
    def from_splitfile(cls, splitfile:str, *a, **kw):
        filepairs = datalib.load_file_pairs(splitfile)
        return cls(filepairs, *a, **kw)

    
    def _preprocess_treeringmaps(
        self, 
        filepairs: tp.List[tp.Tuple[str,str]], 
        inputsize: int,
        dilation:  int = 0,
    ):
        '''Normalize annotations: 
           resize to input size, skeletonize, dilate to common width'''
        new_cachefile = os.path.join(self.cachedir, 'cachefile2.csv')
        if os.path.exists(new_cachefile):
            return datalib.load_file_pairs(new_cachefile)

        targetfiles = [anf for _,anf in filepairs]
        cachedir = os.path.join(self.cachedir, 'targets')

        new_target_patches: tp.List[str] = []
        for i, anf in enumerate(targetfiles):
            basename = os.path.basename(anf)
            an = datalib.load_image(anf, mode='L') > 0.5
            if an.shape[-2] > inputsize *2 or an.shape[-1] > inputsize *2:
                # intermediate resize to avoid losing thin lines
                intermediate_size = (
                    int( (an.shape[-2] + inputsize)/2 ), 
                    int( (an.shape[-1] + inputsize)/2 )
                )
                an = datalib.resize_tensor(an.float(), intermediate_size, mode='bilinear')
            an = datalib.resize_tensor(an.float(), size=inputsize, mode='bilinear')
            an = (an > 0.0)[0]
            skeleton = skmorph.skeletonize(an.numpy())
            if dilation > 0:
                skeleton = skmorph.dilation(skeleton, skmorph.disk(dilation))
            
            targetfile = os.path.join(cachedir, basename+f'.skel.png')
            datalib.write_image_tensor(targetfile, torch.as_tensor(skeleton))
            new_target_patches.append(targetfile)
        
        # sanity check
        assert len(new_target_patches) == len(self)
    
        input_patches = [inf for inf,_ in self.filepairs]
        new_items     = list( zip(input_patches, new_target_patches) )
        datalib.save_file_tuples(new_cachefile, new_items)

        return new_items
    
    def _resize_inputs_to_inputsize(self, inputsize:int):
        inputfiles = [inf for inf,_ in self.items]
        for i, inf in enumerate(inputfiles):
            x = datalib.load_image(inf, mode='RGB')
            x = datalib.resize_tensor(x, size=inputsize, mode='bilinear')
            datalib.write_image_tensor(inf, x)

    
    def __getitem__(self, i:int):
        inputfile, targetfile = self.items[i]
        input  = datalib.load_image(inputfile)
        target = datalib.load_image(targetfile, mode='L')
        return input, target

    def collate_fn(self, raw_batch: RawBatch):
        return raw_batch



#TODO: merge with cells model
class TreeringsInference(torch.nn.Module):
    def __init__(self, module:torch.nn.Module, patchsize:int):
        super().__init__()
        self.module = module
        self.patchsize = patchsize
        self.slack = 32                                                         # TODO

        self._device_indicator = torch.nn.Parameter(torch.empty(0))

    # NOCMMIT: remove?
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




