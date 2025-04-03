import typing as tp


import numpy as np
from skimage import measure as skmeasure
import torch
import torchvision

from traininglib import datalib
from traininglib.segmentation import PatchedCachingDataset, load_if_cached
from traininglib.segmentation.connectedcomponents import (
    connected_components_from_adjacency_list,
    _relabel,
)
from traininglib.modellib import BaseModel
from traininglib.trainingtask import TrainingTask, Loss, Metrics

from .util import load_and_scale_image


Box = tp.Tuple[int,int,int,int]



# assuming 15-150um cell sizes, this results in 15-150px
# TODO: automatically estimate from ground truth
HARDCODED_GOOD_RESOLUTION = 1000   # px/mm

# 0.8mm ~ 800px patch size
HARDCODED_GOOD_PATCHSIZE_MM = 0.1 * 8

# 75um ~ 75px slack
HARDCODED_GOOD_SLACK_MM = 0.075


def resolution_to_scale(px_per_mm:float) -> float:
    return  HARDCODED_GOOD_RESOLUTION / px_per_mm



class CellDetectionModel(BaseModel):
    def __init__(self, *a, px_per_mm:float, **kw):
        basemodule = CellDetectionModule()
        patchsize  = int(HARDCODED_GOOD_PATCHSIZE_MM * HARDCODED_GOOD_RESOLUTION)
        super().__init__(*a, module=basemodule, inputsize=patchsize, **kw)
        self.px_per_mm = px_per_mm
        self.scale     = resolution_to_scale(px_per_mm)
        self.patchify  = True
        self.slack = 2** int(
            np.round(np.log2(HARDCODED_GOOD_SLACK_MM * HARDCODED_GOOD_RESOLUTION))
        )
    
    def start_training(self, *a, task_kw = {}, fit_kw = {}, **kw):
        task_kw = {
            'scale': self.scale,
            #'inputsize': self.inputsize,
        } | task_kw
        fit_kw = {'lr': 1e-4} | fit_kw
        return super()._start_training(
            CellsTrainingTask, 
            *a, 
            task_kw = task_kw, 
            fit_kw  = fit_kw, 
            **kw
        )
    
    # TODO: code re-use with treerings
    def prepare_image(self, image:str|np.ndarray):
        # NOTE: no normalize because of memory issues
        # TODO: use tifffile, read patch, resize, repeat!
        if isinstance(image, str):
            #image = datalib.load_image(image, to_tensor=True, normalize=False)
            image = load_and_scale_image(image, self.scale)  # type: ignore
        
        x = x0 = torch.as_tensor(image)
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
        raw: tp.List[tp.List[tp.Dict[str,torch.Tensor]]],
        x:   torch.Tensor,
    ) -> tp.Dict[str, np.ndarray]:
        raw_ = [{k:v.cpu() for k,v in y[0].items()} for y in raw]

        assert self.patchify, NotImplemented
        if self.patchify:
            instancemaps = [output['instances'] for output in raw_]
            grid = datalib.grid_for_patches(x.shape[-2:], self.inputsize, self.slack)
            instancemap = \
                stitch_and_relabel_instancemaps_from_grid(instancemaps, grid, self.slack)
        instancepoints = instancemap_to_points(instancemap.numpy(), 1/self.scale)
        return {
            'instancemap': instancemap.numpy(),
            'cell_points': instancepoints,     # type: ignore
            'classmap':    (instancemap > 0).numpy(),
        }


def instancemap_to_points(
    instancemap: np.ndarray, 
    scale:       float = 1.0
) -> tp.List[np.ndarray]:
    assert instancemap.ndim == 2 and instancemap.dtype in [np.int64, np.int32]

    contours:tp.List[np.ndarray] = []
    for i, prop in enumerate(skmeasure.regionprops(instancemap), start=1):
        propslice:tp.Tuple[slice, slice] = prop.slice
        topleft_xy = propslice[1].start, propslice[0].start
        instancemask = prop.image_convex
        # pad to make sure there are zeros at the borders
        # otherwise will get multiple contours
        instancemask = np.pad(instancemask, pad_width=1)
        contour_yx = skmeasure.find_contours(instancemask, fully_connected='high')
        contour_xy = np.array(contour_yx[0])[...,::-1]
        contour_xy = contour_xy + topleft_xy -1 # -1 because of np.pad
        contour_xy = contour_xy * scale
        contours.append(contour_xy)
    return contours




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
    adjacency_labels  = torch.cat( [overlap_uniques[:,:1], overlap_uniques[:,:1]], dim=-1 )
    relabeled_map1    = _relabel(map1, overlap_uniques, adjacency_labels)
    unchanged_in_map1 = (map1 == relabeled_map1)
    relabeled_map1 = torch.where( 
        (map1 > 0) & unchanged_in_map1, 
        relabeled_map1 + map0.max() + 1, 
        relabeled_map1 
    )
    return relabeled_map1

def stitch_and_relabel_instancemaps_from_grid(
    instancemaps: tp.List[torch.Tensor], 
    grid:         torch.Tensor,
    slack:        int,
) -> torch.Tensor:
    assert grid.ndim == 3 and grid.shape[-1] == 4
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
            overlap0  = (x0 - gridcell0[1], y0 - gridcell0[0], w, h)
            overlap1  = (x0 - gridcell1[1], y0 - gridcell1[0], w, h)
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
            overlap0  = (x0 - gridcell0[1], y0 - gridcell0[0], w, h)
            overlap1  = (x0 - gridcell1[1], y0 - gridcell1[0], w, h)
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


class CellsDataset:
    def __init__(self, filepairs:tp.List[tp.Tuple[str,str]], scale:float):
        self.filepairs = filepairs
        self.scale = scale

    # def _cache(self, filepairs:tp.List[tp.Tuple[str,str]]):
    #     for inputfile, annotationfile in filepairs:
    #         load_annotationfile(annotationfile)

    def __len__(self) -> int:
        return len(self.filepairs)

    def __getitem__(self, i:int) -> tp.Tuple[torch.Tensor, tp.Dict]:
        inputfile, annotationfile = self.filepairs[i]
        inputdata = datalib.load_image(inputfile, to_tensor=True)
        inputdata = scale_imagetensor(inputdata, self.scale, 'bilinear')
        masks, boxes = load_annotationfile(annotationfile, self.scale)
        
        targets = {
            'boxes': boxes,
            'masks': masks,
        }
        return inputdata, targets
    
    def collate_fn(self, x:tp.List):
        return x


def scale_imagetensor(x:torch.Tensor, scale:float, mode:str) -> torch.Tensor:
    assert x.ndim in [2,3,4]
    H,W = x.shape[-2], x.shape[-1]
    newshape = (int(H*scale), int(W*scale))
    x = datalib.resize_tensor(x, newshape, mode)
    return x


def load_annotationfile(path:str, scale:float):
    '''Load binary mask and '''
    mask = datalib.load_image(path, to_tensor=True, mode='L')

    labeled  = skmeasure.label(mask[0] > 0)
    regions  = skmeasure.regionprops(labeled)
    boxes_yx = torch.tensor([r.bbox for r in regions]).reshape(-1,4)
    boxes_xy = boxes_yx[..., (1,0,3,2)]
    boxes_xy = boxes_xy * scale
    
    # NOTE: resizing for speed
    # NOTE: resize after connected components
    # otherwise two objects might get too close and counted as one
    labeled = torch.as_tensor(labeled).float()
    labeled = scale_imagetensor(labeled[None], scale, mode='nearest')
    masks = torch.as_tensor( 
        (labeled == torch.unique(labeled)[1:,np.newaxis,np.newaxis]) 
    )
    return masks, boxes_xy



# TODO: custom target resolution
class CellDetectionModule(torch.nn.Module):
    def __init__(self, scale:float=1.0, patchsize:int=800):
        super().__init__()
        self.basemodule = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained = True, 
            progress   = False,
        )
        self.basemodule.transform.min_size = (patchsize,)
        self.basemodule.roi_heads.score_thresh = 0.5
        self.patchsize = patchsize
        self.scale = scale
        self.slack = 2** int(np.round(np.log2( patchsize*scale*0.15 )))
        print('TODO: adjust maskrcnn parameters')
        #self.basemodule.rpn._pre_nms_top_n['testing']  = max(2000, ds_train.objects_per_patch*40)
        #self.basemodule.rpn._post_nms_top_n['testing'] = 
        #self.basemodule.roi_heads.detections_per_img   = max(100,  ds_train.objects_per_patch*4)

    def forward(self, *x):
        outputs = self.basemodule(*x)
        if self.training:
            return outputs
        for output in outputs:
            instancemaps = masks_to_instancemap(output['masks'])
            output['instances'] = instancemaps
            del output['masks']
        return outputs

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


class CellsTrainingTask(TrainingTask):
    def __init__(self, *args, scale:float, **kwargs):
        super().__init__(*args,  **kwargs)
        self.patchsize  = self.basemodule.inputsize
        self.scale = scale
        
    def training_step(self, batch:tp.List) -> tp.Tuple[Loss, Metrics]:
        x,t = prepare_batch(batch, self.patchsize, augment=True, device=self.device)
        lossdict = self.basemodule(x, t)
        loss = torch.stack( [*lossdict.values()] ).sum()
        logs = {k:v.item()  for k,v in lossdict.items()}
        return loss, logs
    
    def create_dataloaders(
        self, 
        trainsplit:tp.List, 
        valsplit = 'ignored', 
        **kw
    ) -> tp.Tuple[tp.Iterable, tp.Iterable|None]:
        ds_train = CellsDataset(trainsplit, self.scale)
        ld_train = datalib.create_dataloader(ds_train, shuffle=True, **kw)

        return ld_train, None


def prepare_batch(
    rawbatch: tp.List, 
    patchsize:int,
    augment:  bool,
    device:   torch.device,
) -> tp.Tuple:
    images = [im for im, _ in rawbatch]
    masks  = [an['masks'] for _, an in rawbatch]
    boxes  = [an['boxes'] for _, an in rawbatch]

    inputs  = []
    targets = []

    for inputimage, mask, boxes_i in zip(images, masks, boxes):
        if augment:
            inputimage, mask, boxes_i = \
                augment_image_and_boxes(inputimage, mask, boxes_i, patchsize)

        labels = torch.ones(len(boxes_i), device=device, dtype=torch.int64)
        inputs.append(inputimage.to(device))
        targets.append({
            'boxes':  boxes_i.to(device),
            'masks':  mask.to(device),
            'labels': labels,
        })
    
    return inputs, targets


def augment_image_and_boxes(
    image:torch.Tensor, 
    masks:torch.Tensor, 
    boxes:torch.Tensor,
    patchsize: int,
):
    image, masks, boxes = \
        random_crop_image_and_boxes(image, masks, boxes, patchsize)
    # TODO: rotate
    # TODO: flip
    # TODO: color jitter
    return image, masks, boxes


def random_crop_image_and_boxes(
    image:torch.Tensor, 
    masks:torch.Tensor, 
    boxes:torch.Tensor, 
    patchsize:int,
    cropfactors:tp.Tuple[float, float] = (0.75, 1.33),
):
    assert image.ndim == 3 and image.shape[0] == 3
    assert masks.ndim == 3 and masks.shape[-2:] == image.shape[-2:]
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    assert len(boxes) == len(masks)

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
    bilinear = datalib.interpolation_modes['bilinear']
    nearest  = datalib.interpolation_modes['nearest']
    image    = resized_crop(image[None], y0, x0, h, w, newsize, bilinear)[0]
    masks    = resized_crop(masks[None], y0, x0, h, w, newsize, nearest)[0]
    
    new_boxes, ok_boxes = resized_crop_on_boxes(boxes, cropbox, newsize)
    new_boxes = new_boxes[ok_boxes]
    masks     = masks[ok_boxes]
    return image, masks, new_boxes


def resized_crop_on_boxes(
    boxes:   torch.Tensor, 
    cropbox: Box,
    new_size:tp.Tuple[int,int],
    minimum_area:float = 0.5,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    assert new_size[0] == new_size[1], NotImplemented
    size = new_size[0]

    new_boxes = torch.cat([
        datalib.adjust_coordinates_for_crop(boxes[..., :2], cropbox, new_size),
        datalib.adjust_coordinates_for_crop(boxes[..., 2:], cropbox, new_size),
    ], dim=-1)
    clipped_boxes = new_boxes.clip(0, size)
    
    areas_og      = box_area(new_boxes)
    areas_clipped = box_area(clipped_boxes)
    ratios        = areas_clipped / areas_og
    ok_boxes      = (ratios >= minimum_area)
    return clipped_boxes, ok_boxes

def box_area(boxes:torch.Tensor) -> torch.Tensor:
    assert boxes.ndim ==2 and boxes.shape[-1] == 4
    x0,y0, x1,y1 = boxes[:,0],boxes[:,1], boxes[:,2],boxes[:,3]
    w = x1 - x0
    h = y1 - y0
    return w * h







if __name__ == '__main__':
    m = CellDetectionModel(inputsize=800, px_per_mm=2800)
    filepairs = [
        ('./UPLOAD/WOODB_3_4_a.png', './UPLOAD/WOODB_3_4_a.cells.png')
    ]
    m.start_training(filepairs)
    m.save('DEBUG/model.DELETE.pt.zip')
    print('done')




if __name__ == 'XXX__main__':
    filepairs = [
        ('./UPLOAD/WOODB_3_4_a.png', './UPLOAD/WOODB_3_4_a.cells.png')
    ]
    import time
    t0 = time.time()
    ds = CellsDataset(filepairs, scale=1/2.8)
    ld = datalib.create_dataloader(ds, batch_size=8)
    batch = next(iter(ld))
    t1 = time.time()
    x,t   = prepare_batch(batch, patchsize=512, augment=True, device='cpu') # type: ignore
    t2 = time.time()
    print(t1-t0, t2-t1)

    b = t[0]['boxes']
    m = t[0]['masks']

    import PIL.Image
    PIL.Image.fromarray(
        (m.any(0).numpy() * 255).astype('uint8')
    ).save('DEBUG/m.DELETE.png')
    x[0] = x[0] * 0.3
    x[0][:, b[:,1].long(), b[:,0].long()] = torch.tensor([0,0,1])[:,None].float()
    x[0][:, b[:,3].long()-1, b[:,2].long()-1] = torch.tensor([1,0,0])[:,None].float()
    PIL.Image.fromarray(
        (x[0].permute(1,2,0).numpy() * 255).astype('uint8')
    ).save('DEBUG/x.DELETE.png')

    print('done')

