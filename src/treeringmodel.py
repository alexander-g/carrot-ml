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
from .cellsmodel import instancemap_to_points

import skimage.measure as skmeasure

from .util import load_and_scale_image



# assuming minimum 0.1mm tree ring width, this results in 25px, should be enough
HARDCODED_GOOD_RESOLUTION = 250   # px/mm

def resolution_to_scale(px_per_mm:float) -> float:
    return  HARDCODED_GOOD_RESOLUTION / px_per_mm



class TreeringDetectionModel(SegmentationModel):
    def __init__(self, *a, px_per_mm:float, **kw):
        kw['patchify'] = True
        super().__init__(
            *a, 
            classes  = [Class('ring-boundaries', (255,255,255))], 
            **kw
        )
        self.px_per_mm = px_per_mm
        self.scale = resolution_to_scale(px_per_mm)
    
    # TODO: profiling, cpu faster than cuda
    def process_image(self, *a, progress_callback='ignored', **kw):
        return super().process_image(*a,  **kw)
    
    def prepare_image(self, image:str|np.ndarray):
        # NOTE: no normalize because of memory issues
        # TODO: use tifffile, read patch, resize, repeat
        if isinstance(image, str):
            #image = datalib.load_image(image, to_tensor=True, normalize=False)
            image = load_and_scale_image(image, self.scale)   # type: ignore
        
        x = x0 = torch.as_tensor(image)
        # if self.scale != 1:
        #     H,W = x.shape[-2:]
        #     newshape = [ int(H * self.scale), int(W * self.scale) ]
        #     x = datalib.resize_tensor(x, newshape, 'bilinear')
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
        return self.segmentation_to_points(segmentation)
    
    def start_training(self, *a, task_kw={}, **kw):
        task_kw = {
            'scale':     self.scale,
            'inputsize': self.inputsize,
        } | task_kw
        return super()._start_training(
            TreeringDetectionTrainingTask, *a, task_kw=task_kw, **kw
        )
    
    def segmentation_to_points(self, segmentation:np.ndarray) -> tp.Dict:
        paths = treeringlib.tree_ring_clustering(segmentation)
        # scale up the paths to original size
        paths = [ p / self.scale for p in paths ]
        ring_labels = treeringlib.associate_boundaries(paths)
        ring_points = [
            treeringlib.associate_pathpoints(paths[r0-1], paths[r1-1]) 
                for r0,r1 in ring_labels
        ]
        ring_areas = [treeringlib.treering_area(*rp) for rp in ring_points]
        return {
            'segmentation'   : segmentation,
            'ring_points'    : ring_points,
            'ring_labels'    : ring_labels,
            'ring_areas'     : ring_areas,
        }
    
    @classmethod
    def associate_cells_from_segmentation(
        cls, 
        cell_map:    np.ndarray, 
        ring_points: tp.List[tp.Tuple[np.ndarray, np.ndarray]], 
        og_size:     tp.Optional[tp.Tuple[int,int]] = None,
    ) -> tp.Tuple:
        '''Assign cells to rings. 
          If og_size is not None will scale the cell coordinates (but not rings)'''
        #return treeringlib.associate_cells_from_segmentation(cell_map, ring_points, og_size)
        cell_map = (cell_map > 0)
        cell_points, instancemap = classmap_to_cell_points(cell_map, og_size)
        return cls.associate_cells(cell_points, ring_points, instancemap)

    @staticmethod
    def associate_cells(
        cell_points: tp.List[np.ndarray], 
        ring_points: tp.List[tp.Tuple[np.ndarray, np.ndarray]], 
        instancemap: np.ndarray,
    ):
        # NOTE: ring_points in yx format, for legacy reasons
        # converting to xy
        ring_points = [
            (p0[...,::-1], p1[...,::-1]) for p0,p1 in ring_points
        ]
        cell_info = associate_cells(cell_points, ring_points)
        ring_per_cell = np.array([info['year'] for info in cell_info])
        ringmap_rgb   = colorize_instancemap(instancemap, ring_per_cell)
        return cell_info, ringmap_rgb


def classmap_to_cell_points(
    classmap: np.ndarray, 
    og_size:  tp.Optional[tp.Tuple[int,int]] = None,
) -> tp.Tuple[tp.List[np.ndarray], np.ndarray]:
    assert classmap.dtype == np.bool_ and classmap.ndim == 2

    instancemap = skmeasure.label(classmap).astype(np.int64)
    cell_points = instancemap_to_points(instancemap)
    if og_size is not None:
        og_scale = np.array([
            classmap.shape[1] / og_size[0],
            classmap.shape[0] / og_size[1],
        ])
        cell_points = [
            p / og_scale for p in cell_points
        ]
    return cell_points, instancemap

def associate_cells(
    cell_points: tp.List[np.ndarray], 
    ring_points: tp.List[tp.Tuple[np.ndarray, np.ndarray]], 
) -> tp.List[tp.Dict]:
    '''Assign cells to a ring. All coordinates in xy format.'''
    cell_indices = np.cumsum([len(cell) for cell in cell_points])
    all_cell_points = np.concatenate(cell_points)
    # which point is in which ring
    ring_for_points = -np.ones(len(all_cell_points), dtype=np.int64)

    for i,(p0,p1) in enumerate(ring_points):
        polygon = np.concatenate([p0,p1[::-1]], axis=0)
        polygon = skmeasure.approximate_polygon(polygon, tolerance=5)

        in_poly_mask = skmeasure.points_in_poly(all_cell_points, polygon)
        ring_for_points = np.where(in_poly_mask, i, ring_for_points)
    
    cellinfo = []
    for j, ring_ixs in enumerate(np.split(ring_for_points, cell_indices)[:-1]):
        uniques, counts = np.unique(ring_ixs, return_counts=True)
        box_xy = [
            cell_points[j][:,0].min(),
            cell_points[j][:,1].min(),
            cell_points[j][:,0].max(),
            cell_points[j][:,1].max(),
        ]

        cellinfo.append({
            'id':              j,
            'box_xy':          box_xy,
            'year':            int(uniques[counts.argmax()]),
            'area':            polygon_area(cell_points[j]),
            'position_within': 0.0,  # TODO
        })
    return cellinfo

def polygon_area(points: np.ndarray) -> float:
    '''Compute the area of a polygon given its vertices ordered clockwise.
       Shoelace formula.'''
    assert points.ndim == 2 and points.shape[1] == 2
    
    shifted_points = np.roll(points, -1, axis=0)
    # cross product components (determinants)
    cross_products = (points[:, 0] * shifted_points[:, 1] -
                      points[:, 1] * shifted_points[:, 0])
    
    area = 0.5 * np.abs(np.sum(cross_products))
    return area


def colorize_instancemap(instancemap:np.ndarray, ring_per_cell:np.ndarray):
    assert instancemap.ndim == 2 and instancemap.dtype == np.int64
    assert ring_per_cell.ndim == 1 and ring_per_cell.dtype == np.int64

    COLORS = [
        #(255,255,255),
        ( 23,190,207),
        (255,127, 14),
        ( 44,160, 44),
        (214, 39, 40),
        (148,103,189),
        (140, 86, 75),
        (188,189, 34),
        (227,119,194),
    ]

    rgb = np.zeros(instancemap.shape+(3,), dtype=np.uint8)
    for i, ring_idx in enumerate(np.unique(ring_per_cell)):
        if ring_idx < 0:
            continue
        cell_ixs  = np.argwhere(ring_per_cell == ring_idx).ravel() +1
        cell_mask = np.isin(instancemap, cell_ixs)
        rgb[cell_mask] = COLORS[i % len(COLORS)]
    return rgb



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

