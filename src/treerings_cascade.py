import typing as tp

import torch

from traininglib.modellib import BaseModel, SaveableModule
from traininglib import unet
from traininglib.segmentation import (
    margin_loss_fn,
    margin_loss_multilabel,
)

from .treeringmodel import HARDCODED_GOOD_RESOLUTION, TreeringsDataset
from .cc_celldetection import prepare_batch







class Cascade_TreeringsModule(torch.nn.Module):
    '''Two sequential UNets. The second one corrects mistakes of the first one.'''
    def __init__(self):
        super().__init__()
        self.stage0 = unet.UNet(output_channels=1)
        self.stage1 = unet.UNet(input_channels=3+1, output_channels=1)
        self.px_per_mm = HARDCODED_GOOD_RESOLUTION
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y0 = self.stage0(x)
        
        x  = torch.cat( [x, y0.detach().sigmoid()], dim=1 )
        y1 = self.stage1(x)

        return torch.cat([y1, y0], dim=1)



RawBatch = tp.List[ tp.Tuple[torch.Tensor, torch.Tensor] ]

class CascadeTrainStep(SaveableModule):
    def __init__(self, module:Cascade_TreeringsModule, inputsize:int):
        super().__init__()
        self.module    = module
        self.inputsize = inputsize
        self._device_indicator = torch.nn.Parameter(torch.empty(0))

    def forward(self, raw_batch:RawBatch):
        x,t_stage0 = prepare_batch(
            raw_batch,
            augment   = True,
            patchsize = self.inputsize,
            device    = self._device_indicator.device
        )
        y:torch.Tensor = self.module(x)
        assert y.shape[1] == 2

        # reverse order
        y0 = y[:,1][:,None]
        y1 = y[:,0][:,None]
        yx = y0.sigmoid() + y1.sigmoid()

        t_stage1 = estimate_mistakes(t_stage0.bool(), (y0.detach() > 0))


        bce_fn = torch.nn.functional.binary_cross_entropy_with_logits
        
        bce0    = bce_fn(y0, t_stage0)
        mgn0    = margin_loss_multilabel(y0, t_stage0.bool(), logits=True)
        loss0   = bce0 + mgn0 * 0.1

        bce1    = bce_fn(y1, t_stage1)
        mgn1    = margin_loss_multilabel(y1, t_stage1.bool(), logits=True)
        loss1   = bce1 + mgn1 * 0.1

        loss = loss0 + loss1

        recall0 = (y0 > 0.0)[t_stage0 > 0].float().mean()
        recall1 = (y1 > 0.0)[t_stage1 > 0].float().mean()
        recallx = (yx > 0.5)[t_stage0 > 0].float().mean()

        
        logs = { 
            'loss0': float(loss0), 
            'loss1': float(loss1),  
            'rec0':  float(recall0), 
            'rec1':  float(recall1),
            'recx':  float(recallx),
        }
        return loss, logs



def pad_to_kernelsize(x:torch.Tensor, k:int):
    W = x.shape[-2]
    H = x.shape[-1]

    pad_x = k - (W % k)
    pad_y = k - (H % k)

    x = torch.nn.functional.pad(x, [0, pad_y, 0, pad_x])
    return x


def dilate_mask(x:torch.Tensor, k:int) -> torch.Tensor:
    '''Fast and dirty dilation via max pooling'''
    assert x.dtype == torch.bool
    H = x.shape[-2]
    W = x.shape[-1]

    x = torch.nn.functional.max_pool2d(
        pad_to_kernelsize(x, k).float(),
        kernel_size = k, 
        stride = k, 
    )
    x = torch.nn.functional.interpolate(x, scale_factor=k, mode='nearest')
    x = x[..., :H, :W].bool()
    return x



def estimate_mistakes(annotations:torch.Tensor, outputs:torch.Tensor):
    assert annotations.dtype == torch.bool
    assert outputs.dtype == torch.bool

    outputs_dilated = dilate_mask(outputs, k=40)
    mistakes = annotations & (~outputs_dilated)
    return mistakes.float()






if __name__ == '__main__':
    m = Cascade_TreeringsModule()
    step = CascadeTrainStep(m, 512)

    x = torch.rand([4,3,1024,1024])
    t = torch.zeros([4,1,1024,1024], dtype=torch.bool)
    t[1,0, 50:900, 200] = 1
    t[1,0, 50:900, 800] = 1

    batch = list(zip(x,t))

    import time
    
    t0 = time.time()
    loss, logs = step(batch)
    t1 = time.time()
    
    print(logs)
    print(t1-t0)

