import os
import typing as tp


import numpy as np
import scipy
import torch
import torchvision

from traininglib import datalib, modellib
from traininglib.segmentation.connectedcomponents import _relabel
from traininglib.segmentation import (
    grid_for_patches, 
    paste_patch, 
    get_patch_from_grid,
)
from traininglib import trainingloop
from .cc_celldetection import CC_CellsDataset
from .cc_postprocessing import delineate_instancemap
from .util import load_and_scale_image



# assuming 10-250um cell sizes, this results in 10-250px
HARDCODED_GOOD_RESOLUTION = 1000   # px/mm

HARDCODED_MIN_CELLSIZE_UM = 10
HARDCODED_MIN_CELLSIZE_PX = \
    HARDCODED_MIN_CELLSIZE_UM * 1000 // HARDCODED_GOOD_RESOLUTION



HARDCODED_DEFAULT_PATCHSIZE = 800













# basic mask rcnn
class MaskRCNN_CellsModule(torch.nn.Module):
    def __init__(self, inputsize:int, target_px_per_mm:tp.Optional[float], **kw):
        super().__init__()
        self.basemodule = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained = True, 
            progress   = False,
            box_detections_per_img = 250,
            box_nms_thresh = 0.30,
        )
        self.basemodule.transform.min_size = (inputsize,)
        self.basemodule.roi_heads.score_thresh = 0.5

        self.inputsize = inputsize
        self.px_per_mm = target_px_per_mm or HARDCODED_GOOD_RESOLUTION

    def forward(self, *x):
        outputs = self.basemodule(*x)
        if self.training:
            return outputs
        for output in outputs:
            instancemaps = masks_to_instancemap(output['masks'])
            output['instances'] = instancemaps
            del output['masks']
        return outputs


RawBatch = tp.List[ tp.Tuple[torch.Tensor, torch.Tensor] ]

class MaskRCNN_TrainStep(modellib.SaveableModule):
    def __init__(self, module:MaskRCNN_CellsModule, inputsize:int):
        super().__init__()
        self.module    = module
        self.inputsize = inputsize
        self._device_indicator = torch.nn.Parameter(torch.empty(0))

    def forward(self, raw_batch:RawBatch):
        x,t = prepare_batch(
            raw_batch,
            augment   = True,
            patchsize = self.inputsize,
            device    = self._device_indicator.device,
            whiteout_prob = 0.025,
        )
        lossdict = self.module(x, t)
        loss = torch.stack( [*lossdict.values()] ).sum()
        logs = {k:v.item()  for k,v in lossdict.items()}
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
    whiteout_prob: float = 0.0,
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

        if augment:
            x, t = datalib.random_crop(
                x[None], 
                t[None],
                patchsize=patchsize, 
                modes=['bilinear', 'nearest']
            )
            x, t = datalib.random_rotate_flip(x, t)

        instancemap = t[0,0]
        # DONT: instancemap = exclude_border_instances(instancemap)

        masks  = instancemap_to_masks(instancemap)
        boxes  = masks_to_boxes(masks)
        labels = torch.ones(len(boxes), dtype=torch.int64)

        all_inputs.append(x[0])
        all_targets.append({
            'boxes':  boxes.to(device),
            'masks':  masks.to(device),
            'labels': labels.to(device),
        })

    inputs = torch.stack(all_inputs).to(device)
    if augment:
        # NOTE: jitter should be done on gpu, slow otherwise
        inputs = jitter(inputs)
    return inputs, all_targets


def instancemap_to_masks(instancemap:torch.Tensor) -> torch.Tensor:
    assert instancemap.ndim == 2
    assert instancemap.dtype == torch.int64
    # relabel to contiguous 0..N
    _, instancemap = torch.unique(instancemap, return_inverse=True)

    masks = torch.nn.functional.one_hot(instancemap).permute(2,0,1)[1:].bool()
    return masks

def masks_to_boxes(masks:torch.Tensor) -> torch.Tensor:
    assert masks.ndim == 3   # CHW
    assert masks.dtype == torch.bool

    all_boxes:tp.List[torch.Tensor] = []
    for i in range(len(masks)):
        coordinates_yx = torch.argwhere( masks[i] )
        coordinates_xy = torch.flip(coordinates_yx, dims=[-1])
        p0 = coordinates_xy.min(0).values
        p1 = coordinates_xy.max(0).values +1
        box = torch.cat([p0,p1])
        all_boxes.append(box)
    
    if len(all_boxes) == 0:
        return torch.empty([0,4])
    return torch.stack(all_boxes).float()


def exclude_border_instances(instancemap:torch.Tensor) -> torch.Tensor:
    assert instancemap.ndim == 2 and instancemap.dtype == torch.int64
    
    instancemap = instancemap.clone()

    row0 = instancemap[0]
    row1 = instancemap[-1]
    col0 = instancemap[:, 0]
    col1 = instancemap[:,-1]
    for i in torch.unique( torch.cat([row0, row1, col0, col1]) ):
        instancemap[instancemap == i] = 0
    return instancemap



def masks_to_instancemap(masks:torch.Tensor, threshold:float=0.5) -> torch.Tensor:
    '''Convert masks as returned by Mask-RCNN (shape [N,1,H,W]) to a [H,W]
       int32 instancemap, with each instance having a unique value'''
    assert masks.ndim == 4 and masks.shape[1] == 1
    if len(masks) == 0:
        return torch.zeros(masks.shape[2:], device=masks.device, dtype=torch.int32)
    masks = masks[:,0]
    masks = (masks > threshold)
    instancelabels = \
        torch.arange(1, len(masks)+1, device=masks.device, dtype=torch.int32)
    instances = (masks * instancelabels[:,None,None]).max(0)[0]
    return instances



class InstanceDataset(CC_CellsDataset):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self.items = self._postprocess_annotations()
    
    def _postprocess_annotations(self):
        '''Perform connected components on annotations'''
        new_cachefile = os.path.join(self.cachedir, 'cachefile2.csv')
        if os.path.exists(new_cachefile):
            return datalib.load_file_pairs(new_cachefile)

        outputdir = os.path.join(self.cachedir, 'labeled')
        new_items = []
        for inf, anf in self.items:
            an = datalib.load_image(anf, mode='L').bool().numpy()
            instancemap, _ = scipy.ndimage.label(an[0], structure=np.ones([3,3]))
            instancemap_t = torch.as_tensor(instancemap)
            
            outputpath = os.path.join(outputdir, os.path.basename(anf))
            datalib.write_image_tensor(outputpath, instancemap_t)

            new_items.append( (inf, outputpath) )
        
        datalib.save_file_tuples(new_cachefile, new_items)
        return new_items
    
    def __getitem__(self, i:int):
        inputfile, targetfile = self.items[i]
        input  = datalib.load_image(inputfile)
        target = datalib.load_image(targetfile, mode='L', normalize=False).long()
        return input, target


class MaskRCNN_Cells_CARROT(modellib.SaveableModule):
    def __init__(self, module:MaskRCNN_CellsModule):
        super().__init__()
        self.module = module
        self.requires_grad_(False).eval()

        self.patchsize = self.module.inputsize
        self.slack = 128                                                        # TODO
        self._device_indicator = torch.nn.Parameter(torch.empty(0))
    
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
        scale = self.module.px_per_mm / px_per_mm
        x, og_shape = load_and_scale_image(imagepath, scale)
        
        # image already scaled down, px_per_mm is now same as module's
        px_per_mm = self.module.px_per_mm
        x, grid, n, _ = self.prepare_full_image(x, px_per_mm)
        instancemap_patches = []
        for i in range(0, n, batchsize):
            instancemaps = self.process_batch(x, grid, i, n, batchsize)
            instancemap_patches.extend(instancemaps)

            if progress_callback is not None:
                progress_callback( i/n )
        instancemap = stitch_and_relabel_instancemaps_from_grid(
            instancemap_patches, 
            grid, 
            self.slack
        )
        classmap = delineate_instancemap(instancemap).cpu().numpy()

        if outputshape is None:
            outputshape = og_shape
        full_output = datalib.resize_tensor2(
            instancemap[None].float(), 
            outputshape, 
            'nearest'
        )[0].to(instancemap.dtype)
        
        return classmap

    
    def prepare_full_image(self, x:torch.Tensor, px_per_mm:float):
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

        # to CHW, not yet to f32 to save memory
        x = x.permute(2,0,1) #/ 255
        x = datalib.resize_tensor2(x, [h,w], 'bilinear' )

        shape = torch.tensor([h,w])
        grid  = grid_for_patches(shape, self.patchsize, self.slack).long()
        n     = grid.reshape(-1,4).shape[0]
        
        return x, grid, n, (H,W)
    
    def prepare_batch(self, x:torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x / 255
        return x
    
    def process_batch(
        self, 
        x:    torch.Tensor, 
        grid: torch.Tensor, 
        i:    int, 
        n:    int,
        batchsize: int
    ) -> tp.List[torch.Tensor]:
        x_patches = []
        for j in range(i, min(i+batchsize, n)):
            gridcell = grid.reshape(-1,4)[j]
            x_patch  = get_patch_from_grid(x, grid, torch.tensor(j))
            x_patches.append(x_patch)

        x_batch = torch.stack(x_patches)
        x_batch = self.prepare_batch(x_batch)
        with torch.no_grad():
            output:torch.Tensor = self.module(x_batch)
        instancemaps = [item['instances'] for item in output]
        return instancemaps





# y0,x0,y1,x1
Box = tp.Tuple[int,int,int,int]


def relabel_instancemaps(
    map0:torch.Tensor, 
    map1:torch.Tensor,
    overlapbox0: Box,
    overlapbox1: Box,
) -> torch.Tensor:
    '''Relabel instance map `map1` so that overlapping instances have the same 
       value as in map0. (overlap boxes in format left,top,width,height) '''
    assert map0.ndim == map1.ndim == 2
    assert map0.dtype in [torch.int64, torch.int32]
    assert map0.dtype == map1.dtype 
    assert overlapbox0[2:] == overlapbox1[2:]
    assert (torch.tensor(overlapbox0) >= 0).all()
    assert (torch.tensor(overlapbox1) >= 0).all()

    x0,y0,w0,h0 = overlapbox0
    x1,y1,w1,h1 = overlapbox1
    overlap0 = map0[y0:, x0:][:h0,:w0]
    overlap1 = map1[y1:, x1:][:h1,:w1]
    assert overlap0.shape == overlap1.shape, [overlap0.shape, overlap1.shape]

    mask = (overlap0 > 0) & (overlap1 > 0)
    overlapping_values = torch.stack([overlap0, overlap1], dim=-1)[mask]
    overlap_uniques, overlap_counts = \
        datalib.faster_unique_dim0_with_counts(overlapping_values)

    # TODO: filter out too-small overlaps
    
    #adjacency_labels  = connected_components_from_adjacency_list(overlap_uniques)
    # TODO: this is a simplification, rework this properly
    adjacency_labels = \
        torch.cat( [overlap_uniques[:,:1], overlap_uniques[:,:1]], dim=-1 )
    relabeled_map1 = relabel(map1, overlap_uniques[:,1], overlap_uniques[:,0])
    #unchanged_in_map1 = (map1 == relabeled_map1)  # wrong!
    unchanged_in_map1 = ~torch.isin(map1, overlap_uniques[:,1])
    relabeled_map1 = torch.where( 
        (map1 > 0) & unchanged_in_map1, 
        relabeled_map1 + map0.max() + 1, 
        relabeled_map1 
    )
    return relabeled_map1

def relabel(
    instancemap: torch.Tensor,
    from_labels: torch.Tensor,
    to_labels:   torch.Tensor,
) -> torch.Tensor:
    assert instancemap.ndim == 2 and instancemap.dtype in [torch.int64, torch.int32]
    assert from_labels.ndim == 1 and from_labels.dtype in [torch.int64, torch.int32]
    assert to_labels.ndim == 1   and to_labels.dtype in [torch.int64, torch.int32]

    n = int(instancemap.max())
    new_labels = torch.arange(n+1, dtype=instancemap.dtype, device=instancemap.device)
    new_labels[from_labels] = to_labels
    new_instancemap = new_labels[instancemap]
    return new_instancemap




def stitch_and_relabel_instancemaps_from_grid(
    instancemaps: tp.List[torch.Tensor], 
    grid:         torch.Tensor,
    slack:        int,
) -> torch.Tensor:
    assert grid.ndim == 3 and grid.shape[-1] == 4
    assert grid.dtype == torch.int64
    grid = torch.as_tensor(grid) # because
    H,W  = grid.shape[:2]
    assert H * W == len(instancemaps)

    grid_rows:tp.List[torch.Tensor] = []
    instancemap_rows:tp.List[torch.Tensor] = []
    for i in range(H):
        for j in range(1, W):
            gridcell0 = grid[i,j-1]
            gridcell1 = grid[i,j]
            overlap   = compute_overlap(gridcell0, gridcell1) # type: ignore
            x0,y0,w,h = overlap
            overlap0  = (int(x0 - gridcell0[1]), int(y0 - gridcell0[0]), w, h)
            overlap1  = (int(x0 - gridcell1[1]), int(y0 - gridcell1[0]), w, h)
            instancemap0 = instancemaps[i*W + j-1]
            instancemap1 = instancemaps[i*W + j]
            instancemap1 = \
                relabel_instancemaps(instancemap0, instancemap1, overlap0, overlap1)
            instancemaps[i*W + j] = instancemap1
        
        gridcell_row_i = torch.cat([grid[i,0,:2], grid[i,-1,2:]], dim=-1)
        row_shape = tuple(grid[0,-1,2:])
        instancemap_row_i = \
            datalib.stitch_overlapping_patches(instancemaps[i*W:][:W], row_shape, slack)
        # NOTE: setting previous maps to None to conserve memory
        for _q in range(i*W, (i+1)*W):
            instancemaps[_q] = None  # type: ignore

        if i > 0:
            gridcell0 = grid_rows[i-1]
            gridcell1 = gridcell_row_i
            overlap   = compute_overlap(gridcell0, gridcell1) # type: ignore
            x0,y0,w,h = overlap
            overlap0  = (int(x0 - gridcell0[1]), int(y0 - gridcell0[0]), w, h)
            overlap1  = (int(x0 - gridcell1[1]), int(y0 - gridcell1[0]), w, h)
            instancemap_row_i = relabel_instancemaps(
                instancemap_rows[i-1],
                instancemap_row_i,
                overlap0,
                overlap1,
            )
        instancemap_rows.append(instancemap_row_i)
        grid_rows.append(gridcell_row_i)

    for i in range(H):
        instancemap_row_i = instancemap_rows[i]
        if i > 0:
            rowslack = (grid[i-1,0,2] - grid[i,0,0]) // 2
            instancemap_row_i = instancemap_row_i[rowslack:]
        if i+1 < H:
            rowslack = (grid[i,0,2] - grid[i+1,0,0]) // 2
            instancemap_row_i = instancemap_row_i[:-rowslack]
        instancemap_rows[i] = instancemap_row_i
    instancemap = torch.cat(instancemap_rows, dim=0)
    return instancemap




def compute_overlap(
    box0: Box, 
    box1: Box,
) -> Box:
    y0_0, x0_0, y1_0, x1_0 = box0
    y0_1, x0_1, y1_1, x1_1 = box1

    intersect_x0 = max(x0_0, x0_1)
    intersect_y0 = max(y0_0, y0_1)
    intersect_x1 = min(x1_0, x1_1)
    intersect_y1 = min(y1_0, y1_1)

    overlap_width  = max(0, intersect_x1 - intersect_x0)
    overlap_height = max(0, intersect_y1 - intersect_y0)

    overlap_box = (intersect_x0, intersect_y0, overlap_width, overlap_height)
    return overlap_box


def start_training_from_carrot(
    filepairs: tp.List[tp.Tuple[str,str]],
    cachedir:  str,
    px_per_mm: float,
    epochs:    tp.Optional[int],
    steps:     tp.Optional[int] = None,
    progress_callback: tp.Optional[tp.Callable[[float], None]] = None,
    finetunemodule: tp.Optional[MaskRCNN_CellsModule] = None,
) -> MaskRCNN_Cells_CARROT:
    patchsize = HARDCODED_DEFAULT_PATCHSIZE
    target_px_per_mm = HARDCODED_GOOD_RESOLUTION

    module = MaskRCNN_CellsModule(patchsize, target_px_per_mm)
    if finetunemodule is not None:
        print( module.load_state_dict(finetunemodule.state_dict()) )
    
    trainstep = MaskRCNN_TrainStep(module, inputsize=patchsize)
    # NOTE: *2 because of cropping augmentations
    dataset = InstanceDataset(
        filepairs, 
        patchsize = patchsize*2, 
        px_per_mm = px_per_mm,
        cachedir  = cachedir,
    )

    ld:tp.Sequence = datalib.create_dataloader( # type: ignore
        dataset, 
        batch_size = 8,
        shuffle    = True,
        loader_type = 'threaded',
    )
    trainingloop.train(trainstep, ld, epochs=epochs, steps=steps, progress_callback=progress_callback, lr=1e-4)

    carrotmodel = MaskRCNN_Cells_CARROT(trainstep.module)
    return carrotmodel



