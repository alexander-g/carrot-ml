import typing as tp

import numpy as np
import PIL.Image
import torch
import torchvision

from traininglib import datalib
from traininglib.segmentation import (
    SegmentationModel, 
    PatchwiseTrainingTask,
    Class,
    margin_loss_fn,
)
from traininglib.modellib import BaseModel
from traininglib.paths.pathdataset import FirstComponentPatchedPathsDataset
from traininglib.paths import pathutils as utils
from . import treerings_clustering_legacy as treeringlib



# assuming minimum 0.1mm tree ring width, this results in 100px, should be enough
HARDCODED_GOOD_RESOLUTION = 1000   # px/mm

def resolution_to_scale(px_per_mm:float) -> float:
    return  HARDCODED_GOOD_RESOLUTION / px_per_mm



class TreeringDetectionModel(SegmentationModel):
    def __init__(self, *a, px_per_mm:float, **kw):
        super().__init__(
            *a, 
            classes  = [Class('ring-boundaries', (255,255,255))], 
            patchify = True,
            **kw
        )
        self.px_per_mm = px_per_mm
        self.scale = resolution_to_scale(px_per_mm)
    
    def process_image(self, *a, progress_callback='ignored', **kw):
        return super().process_image(*a,  **kw)
    
    def prepare_image(self, image:str|np.ndarray):
        if isinstance(image, str):
            image = datalib.load_image(image, to_tensor=True, normalize=False)
        
        x = x0 = torch.as_tensor(image)
        if self.scale != 1:
            H,W = x.shape[-2:]
            newshape = [ int(H * self.scale), int(W * self.scale) ]
            x = datalib.resize_tensor(x, newshape, 'bilinear')
        x0 = x    # no resize
        x  = x.float() / 255
        if self.patchify:
            x = datalib.pad_to_minimum_size(x, self.inputsize)
            x = datalib.slice_into_patches_with_overlap(x, self.inputsize, self.slack)
            xbatch = [xi[None] for xi in x]
        else:
            xbatch = [x]
        return xbatch, x0
    
    def finalize_inference(   # type: ignore [override]
        self, 
        raw: tp.List[torch.Tensor], 
        x:   torch.Tensor,
    ):
        segmentation = super().finalize_inference(raw, x).classmap
        paths, points, labels = treeringlib.tree_ring_clustering(segmentation)
        # TODO: need to scale up the paths
        ring_labels  = treeringlib.associate_boundaries(points, labels)
        ring_points  = [
            treeringlib.associate_pathpoints(paths[r0-1], paths[r1-1]) 
                for r0,r1 in ring_labels
        ]
        ring_areas = [treeringlib.treering_area(*rp) for rp in ring_points]
        return {
            'segmentation'   : segmentation,
            'ring_points'    : ring_points,
            'points'         : points,
            'labels'         : labels,
            'ring_labels'    : ring_labels,
            'ring_areas'     : ring_areas,
        }
    
    def start_training(self, *a, task_kw, **kw):
        task_kw = {
            'px_per_mm': self.px_per_mm,
            'inputsize': self.inputsize,
        } | task_kw
        return super()._start_training(
            TreeringDetectionTrainingTask, *a, task_kw=task_kw, **kw
        )
    
    @staticmethod
    def segmentation_to_points(segmentation:np.ndarray) -> tp.Dict:
        paths, points, labels  = treeringlib.tree_ring_clustering(segmentation)
        rings                  = treeringlib.associate_boundaries(points, labels)
        ring_points            = [
            treeringlib.associate_pathpoints(paths[r0-1], paths[r1-1]) for r0,r1 in rings
        ]
        ring_areas             = [treeringlib.treering_area(*rp) for rp in ring_points]
        return {
            'segmentation'   : segmentation,
            'ring_points'    : ring_points,
            'points'         : points,
            'labels'         : labels,
            'ring_labels'    : rings,
            'ring_areas'     : ring_areas,
        }
    
    @staticmethod
    def associate_cells_from_segmentation(cell_map:np.ndarray, ring_points):
        return treeringlib.associate_cells_from_segmentation(cell_map, ring_points)
    


class TreeringDetectionTrainingTask(PatchwiseTrainingTask):
    def __init__(self, *a, scale:float, pos_weight:float, margin_weight:float, **kw):
        super().__init__(*a, **kw)
        self.scale = scale        

    def create_dataloaders(self, *a, **kw) -> tp.Tuple[tp.Iterable, tp.Iterable|None]:
        return self._create_dataloaders(
            FirstComponentPatchedPathsDataset, 
            *a, 
            trainpatchfactor = 2, 
            ds_kw = {'scale': self.scale},
            **kw,
        )

    def training_step(self, raw_batch:tp.List):
        x, t = prepare_batch(raw_batch, 512, augment=True, device=self.device)
        y  = self.basemodule(x)

        bce_fn = torch.nn.functional.binary_cross_entropy_with_logits

        loss_hm_bce = bce_fn(y, t)
        loss_hm_mgn = margin_loss_fn(y, t.bool())  * 0.1

        loss = loss_hm_bce + loss_hm_mgn
        logs = {'bce': float(loss_hm_bce), 'mgn':float(loss_hm_mgn)}
        return loss, logs




def prepare_batch(
    raw_batch: tp.List, 
    patchsize: int,
    augment:   bool, 
    device:    torch.device,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    all_images_raw  = [item[0] for item in raw_batch]
    all_paths_np    = [item[1] for item in raw_batch]

    heatmaptargets = []
    all_images = []

    size = patchsize
    for inputimage, paths_np in zip(all_images_raw, all_paths_np):
        paths = [torch.as_tensor(p).float() for p in paths_np]
        if augment:
            augmented  = augment_image_and_paths(inputimage, paths, size)
            inputimage = augmented['inputimage'] # type: ignore
            paths      = augmented['paths']      # type: ignore
        else:
            assert inputimage.ndim == 3 and inputimage.shape[0] == 3
            h,w = inputimage.shape[-2:]
        all_images.append(inputimage)

        heatmap = utils.rasterize_multiple_paths_batched(
            utils.encode_numpy_paths(paths, 0), 
            n_batches  = 1, 
            size       = size, 
            stepfactor = 2.0,
        ).float()
        heatmaptargets.append(heatmap)
    
    inputs = torch.stack(all_images).to(device)
    targets = torch.stack(heatmaptargets).to(device)
    return inputs, targets



def augment_image_and_paths(
    image: torch.Tensor, 
    paths: tp.List[torch.Tensor],
    patchsize: int,
) -> tp.Dict[str, tp.Union[torch.Tensor, tp.List[torch.Tensor]]]:
    assert image.ndim == 3 and image.shape[0] == 3

    # NOTE: first crop then rotate, for a downstream function
    image, paths, box = random_crop_image_and_paths(image, paths, patchsize)
    k     = int(torch.randint(0,4, [1]))
    image = torch.rot90(image, k, dims=(1,2))
    imshape = image.shape[-2:]
    paths  = [datalib.rot90_coordinates(p, imshape, k) for p in paths]
    jitter = torchvision.transforms.ColorJitter(
        brightness = (0.7, 1.3),
        contrast   = (0.7, 1.3),
        saturation = (0.7, 1.3),
        hue        = (-0.15, 0.15)
    )
    image = jitter(image[None])[0]
    return {
        'inputimage': image,
        'paths':      paths,
        'cropbox':    box,
        'rot90_k':    torch.tensor(k),
    }
    return image, paths


def random_crop_image_and_paths(
    image: torch.Tensor, 
    paths: tp.List[torch.Tensor],
    patchsize:  int,
    cropfactors:tp.Tuple[float, float] = (0.75, 1.33),
) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor], torch.Tensor]:
    assert image.ndim == 3 and image.shape[0] == 3

    H,W = image.shape[-2:]
    lo  = patchsize * cropfactors[0]
    hi  = patchsize * cropfactors[1]
    h   = int(lo + torch.rand(1) * (hi-lo))
    w   = int(lo + torch.rand(1) * (hi-lo))
    y0  = int(torch.rand(1) * (H - h))
    x0  = int(torch.rand(1) * (W - w))

    cropbox = (x0,y0,w,h)
    newsize = (patchsize, patchsize)
    resized_crop = torchvision.transforms.functional.resized_crop
    mode  = datalib.interpolation_modes['bilinear']
    image = resized_crop(image[None], y0, x0, h, w, newsize, mode)[0]
    
    new_paths = []
    for path in paths:
        new_paths.append(
            datalib.adjust_coordinates_for_crop(path, cropbox, newsize)
        )
    return image, new_paths, torch.tensor(cropbox)

