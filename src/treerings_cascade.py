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
PreparedBatch = tp.Tuple[torch.Tensor, torch.Tensor]

class CascadeTrainStep(SaveableModule):
    def __init__(self, module:Cascade_TreeringsModule, inputsize:int):
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
        y:torch.Tensor = self.module(x)
        assert y.shape[1] == 2

        # reverse order
        y0 = y[:,1][:,None]
        y1 = y[:,0][:,None]


        bce_fn = torch.nn.functional.binary_cross_entropy_with_logits
        
        bce0    = bce_fn(y0, t)
        mgn0    = margin_loss_multilabel(y0, t.bool(), logits=True)
        loss0   = bce0 + mgn0 * 0.1

        bce1    = bce_fn(y1, t)
        mgn1    = margin_loss_multilabel(y1, t.bool(), logits=True)
        loss1   = bce1 + mgn1 * 0.1

        loss = loss0 + loss1

        recall0 = (y0 > 0.0)[t > 0].float().mean()
        recall1 = (y1 > 0.0)[t > 0].float().mean()
        
        logs = { 
            'loss0': float(loss0), 
            'loss1': float(loss1),  
            'rec0':  float(recall0), 
            'rec1':  float(recall1) 
        }
        return loss, logs





if __name__ == '__main__':
    m = Cascade_TreeringsModule()
    step = CascadeTrainStep(m, 512)

    x = torch.rand([4,3,1024,1024])
    t = torch.zeros([4,1,1024,1024], dtype=torch.bool)
    t[1,0, 50:900, 200] = 1
    t[1,0, 50:900, 800] = 1

    batch = list(zip(x,t))

    loss, logs = step(batch)
    print(logs)

