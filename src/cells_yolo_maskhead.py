import typing as tp


import torch
import torchvision

from traininglib import datalib, modellib
from traininglib.segmentation import margin_loss_fn
from traininglib.unet import _clone_conv2d_with_new_input_channels

from .maskrcnn_celldetection import (
    InstanceDataset, 
    instancemap_to_masks, 
    masks_to_boxes,
)





class MaskHead(torch.nn.Module):
    '''Very basic segmentation model that crops an image at boxes and segments them.'''

    def __init__(self, masksize:int=64):
        super().__init__()
        self.masksize = masksize
        n_c = masksize ** 2
        # self.module = torchvision.models.shufflenet_v2_x0_5(weights='DEFAULT')
        self.module = torchvision.models.shufflenet_v2_x1_0(weights='DEFAULT')

        n_input_channels = 4
        self.module.conv1[0] =\
            _clone_conv2d_with_new_input_channels(self.module.conv1[0], n_input_channels)

        # future-proofing for myself, in case I change the model
        assert hasattr(self.module, 'fc')
        # replace the last layer
        self.module.fc = torch.nn.Linear(1024, n_c)

    def forward(
        self, 
        x:torch.Tensor, 
        boxes:tp.List[torch.Tensor], 
        # only used during training to indicate presence of other boxes
        all_boxes:tp.Optional[tp.List[torch.Tensor]] = None
    ) -> tp.List[torch.Tensor]:
        assert x.ndim == 4
        assert len(x) == len(boxes)
        if all_boxes is not None:
            assert len(boxes) == len(all_boxes)
        
        all_crops = []
        for i, _boxes in enumerate(boxes):
            # a forth channel that is added to the input
            # so that each crop contains other overlapping boxes
            # to learn not to segment them in crowded areas
            boxcanvas = torch.zeros(x.shape[-2:], device=x.device)
            boxes_to_render = all_boxes[i] if all_boxes is not None else _boxes
            boxcanvas = accumulate_boxes_on_canvas(boxcanvas, boxes_to_render)
            # -1 to remove the self-box
            boxcanvas = boxcanvas - 1

            x_i = x[i]
            x_i = torch.cat([x_i, boxcanvas[None]])
            crops = crop_tensor_at_boxes(x_i, _boxes, self.masksize, 'bilinear')
            all_crops.append(crops)

        all_crops_cat = torch.cat(all_crops)      
        y = self.module(all_crops_cat)
        y = torchscriptable_crop_reshape(y, self.masksize)
        y = torch.split(y, [b.shape[0] for b in boxes])
        return y


@torch.jit.script_if_tracing
def torchscriptable_crop_reshape(x:torch.Tensor, masksize:int) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.empty([0, masksize, masksize], dtype=x.dtype, device=x.device)
    else:
        return x.reshape(x.shape[0], masksize, masksize)




RawBatch = tp.List[ tp.Tuple[torch.Tensor, torch.Tensor] ]

class MaskHeadTrainStep(modellib.SaveableModule):
    def __init__(self, module:torch.nn.Module):
        super().__init__()
        self.module = module
        self._device_indicator = torch.nn.Parameter(torch.empty(0))
    
    def forward(self, raw_batch:RawBatch):
        x, t = prepare_batch(
            raw_batch, 
            augment = True,
            device  = self._device_indicator.device,
        )

        y    = self.module(x['images'], x['boxes'], x['all_boxes'])
        y    = torch.cat(y)
        bce_fn = torch.nn.functional.binary_cross_entropy_with_logits
        bce    = bce_fn(y, t)
        #mgn    = margin_loss_fn(y[:,None], t.bool()[:,None])
        # loss   = bce + mgn * 0.1
        dice   = dice_loss(y, t)
        loss   = bce + dice * 1.0

        accuracy = ((y > 0) == t).float().mean()
        logs = {
            'loss': float(loss),
            'acc':  float(accuracy),
        }
        return loss, logs


def dice_loss(logits:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
    assert logits.ndim == 3
    assert targets.shape == logits.shape

    targets = targets.float()

    probs = torch.sigmoid(logits)
    dims = (1, 2)

    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    dice_loss = 1 - dice.mean()

    return dice_loss



def prepare_batch(
    raw_batch: RawBatch, 
    augment:   bool, 
    device:    torch.device,
    masksize:  int = 64,
    max_masks: int = 1,
):
    all_images  = []
    all_boxes   = []
    all_targets = []
    all_extra_boxes = []

    for i, item in enumerate(raw_batch):
        x = torch.as_tensor(item[0])
        t = torch.as_tensor(item[1])

        if augment:
            # x, t = datalib.random_crop(
            #     x[None], 
            #     t[None],
            #     patchsize=patchsize, 
            #     modes=['bilinear', 'nearest']
            # )
            x, t = datalib.random_rotate_flip(x, t)

        instancemap = t[0]
        assert instancemap.ndim == 2
        # DONT: instancemap = exclude_border_instances(instancemap)

        masks       = instancemap_to_masks(instancemap)  # one-hot masks
        boxes       = masks_to_boxes(masks)
        box_indices = torch.arange(len(boxes), device=boxes.device)
        if augment:
            boxes   = augment_boxes(boxes)
        extra_boxes = boxes
        if len(boxes) > max_masks and augment:
            box_indices = torch.randperm(len(boxes))[:max_masks]
            boxes       = boxes[box_indices]
        targets = instancemap_to_targets(instancemap, boxes, box_indices+1, size=masksize)

        all_images.append(x)
        all_boxes.append(boxes.to(device))
        all_extra_boxes.append(extra_boxes.to(device))
        all_targets.append(targets)

    images  = torch.stack(all_images).to(device)
    targets = torch.cat(all_targets).to(device)
    # if augment:
    #     # NOTE: jitter should be done on gpu, slow otherwise
    #     inputs = jitter(inputs)
    return {'images': images, 'boxes':all_boxes, 'all_boxes':all_extra_boxes}, targets




def crop_tensor_at_boxes(
    x:     torch.Tensor,
    boxes: torch.Tensor,
    size:  int,
    mode:  str,
) -> torch.Tensor:
    assert x.ndim in [2,3]
    assert boxes.ndim == 2
    assert boxes.shape[-1] == 4

    ndim_is_two = (x.ndim == 2)
    if ndim_is_two:
        x = x[None]

    C,H,W    = x.shape
    boxes    = boxes / torch.tensor([W,H,W,H])[None].to(boxes.device) * 2 - 1
    grid     = box_grids(boxes, size)    # [N,n,n,2]
    flatgrid = grid.reshape([1,1,-1,2])  # grid_sample needs batch dim to be same
    flatcrop = torch.nn.functional.grid_sample(x.float()[None], flatgrid, mode=mode, align_corners=True)
    if ndim_is_two:
        crops = flatcrop.reshape(-1, size, size)
    else:
        crops = flatcrop.reshape(C, -1, size, size).permute(1, 0, 2, 3)
    return crops

@torch.jit.script_if_tracing
def convert_boxmasks_into_full_result(
    shape:  tp.Tuple[int,int], 
    boxes:  torch.Tensor, 
    masks:  torch.Tensor,
) -> torch.Tensor:
    '''Cropped output masks to a full [N,H,W] tensor with individual masks'''
    assert boxes.ndim == 2          # [N,4] x0,y0,x1,y1
    assert boxes.shape[1] == 4
    assert masks.ndim == 3          # [N,H,W]
    assert len(boxes) == len(masks)
    
    masks = masks.float()

    H, W = shape
    output = \
        torch.zeros( (boxes.shape[0],H,W), dtype=masks.dtype, device=masks.device )

    x0 = boxes[:,0]
    y0 = boxes[:,1]
    x1 = boxes[:,2]
    y1 = boxes[:,3]

    # integer coordinates in the result that fall inside the box.
    ix0 = torch.ceil(x0).clip(0, None).to(torch.int64)
    iy0 = torch.ceil(y0).clip(0, None).to(torch.int64)
    ix1 = torch.ceil(x1).clip(None, W).to(torch.int64)
    iy1 = torch.ceil(y1).clip(None, H).to(torch.int64)
    intboxes = torch.stack([ix0, iy0, ix1, iy1], dim=1)

    outofbounds = (ix0 >= ix1) | (iy0 >= iy1)
    inbounds    = ~outofbounds
    boxes       = boxes[inbounds]
    intboxes    = intboxes[inbounds]
    masks       = masks[inbounds]

    for i, (box, intbox, mask) in enumerate(zip(boxes, intboxes, masks)):
        # x0, y0, x1, y1 = box   # onnx export error
        x0 = box[0]
        y0 = box[1]
        x1 = box[2]
        y1 = box[3]

        x0int = int(intbox[0])
        y0int = int(intbox[1])
        x1int = int(intbox[2])
        y1int = int(intbox[3])

        # output-space integer coordinates -> mask-space coordinates
        ys = torch.arange(y0int, y1int, device=masks.device, dtype=boxes.dtype)
        xs = torch.arange(x0int, x1int, device=masks.device, dtype=boxes.dtype)

        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        # to -1..+1
        gx = 2 * (xx - x0) / (x1 - x0) - 1
        gy = 2 * (yy - y0) / (y1 - y0) - 1
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)

        sampled = torch.nn.functional.grid_sample(
            mask[None, None].float(),
            grid.float(),
            mode          = "bilinear",
            padding_mode  = "zeros",
            align_corners = False,
        )[0, 0]

        output[i, y0int:y1int, x0int:x1int] += sampled.to(output.dtype)
    return output


    


def scale_boxes(boxes:torch.Tensor, scale:float|torch.Tensor) -> torch.Tensor:
    assert boxes.ndim == 2
    assert boxes.shape[-1] == 4
    if torch.is_tensor(scale):
        assert scale.shape == (len(boxes),) or scale.shape == (len(boxes), 2)
        if scale.ndim == 1:
            scale = torch.stack([scale, scale], dim=-1)

    x0 = boxes[:,0]
    y0 = boxes[:,1]
    x1 = boxes[:,2]
    y1 = boxes[:,3]

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    w  = torch.abs(x1 - x0)
    h  = torch.abs(y1 - y0)

    boxsizes = torch.abs(boxes[:, 2:] - boxes[:, :2])
    boxsizes = boxsizes * scale

    w = boxsizes[:,0]
    h = boxsizes[:,1]

    x0 = cx - w / 2
    y0 = cy - h / 2
    x1 = cx + w / 2
    y1 = cy + h / 2

    boxes = torch.stack([x0,y0,x1,y1], dim=-1)
    return boxes

def augment_boxes(boxes:torch.Tensor) -> torch.Tensor:
    offset = torch.rand([len(boxes),2]) * 10 - 5     # -5 .. 5
    scale  = torch.rand([len(boxes),2]) * 0.5 + 0.8  # 0.8 .. 1.3
    
    boxes  = boxes + torch.cat([offset,offset], dim=-1)
    boxes  = scale_boxes(boxes, scale)
    return boxes



def instancemap_to_targets(
    instancemap: torch.Tensor, 
    boxes:       torch.Tensor, 
    box_indices: torch.Tensor,
    size:        int,
) -> torch.Tensor:
    assert instancemap.ndim == 2
    assert instancemap.dtype == torch.int64
    assert boxes.ndim == 2
    assert boxes.shape[-1] == 4
    assert box_indices.dtype == torch.int64
    assert box_indices.ndim == 1
    assert len(box_indices) == len(boxes)
    assert not (box_indices == 0).any(), \
        'box indices must be incremented by one to match the instancemap values'

    crops   = crop_tensor_at_boxes(instancemap, boxes, size, 'nearest')
    # indices = torch.arange(len(crops), device=crops.device) + 1
    masks   = crops == box_indices[:,None,None]
    return masks.float()


def box_grids(boxes: torch.Tensor, n: int) -> torch.Tensor:
    """Return an [n,n] grid of points inside each xyxy box."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    if n <= 0:
        raise ValueError("n must be positive")

    x1, y1, x2, y2 = boxes.unbind(dim=-1)

    t = torch.linspace(
        0, 1, n, device=boxes.device, dtype=boxes.dtype
    )
    gy, gx = torch.meshgrid(t, t, indexing="ij")

    x = x1[:, None, None] + gx[None] * (x2 - x1)[:, None, None]
    y = y1[:, None, None] + gy[None] * (y2 - y1)[:, None, None]

    return torch.stack((x, y), dim=-1)  # [N, n, n, 2]



def accumulate_boxes_on_canvas(canvas:torch.Tensor, boxes:torch.Tensor) -> torch.Tensor:
    '''Fill out boxes in the canvas, adding up values. 
       i.e a pixel that is inside 3 boxes will have value of 3'''
    y, x = torch.meshgrid(
        torch.arange(canvas.shape[0], device=canvas.device, dtype=torch.int32),
        torch.arange(canvas.shape[1], device=canvas.device, dtype=torch.int32),
        indexing="ij",
    )
    inside = (
          (x >= boxes[:, 0, None, None])
        & (y >= boxes[:, 1, None, None])
        & (x <  boxes[:, 2, None, None])
        & (y <  boxes[:, 3, None, None])
    )
    return canvas + inside.sum(dim=0).to(canvas.dtype)
