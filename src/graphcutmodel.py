import typing as tp

import torch
from traininglib.unet import UNet
from traininglib.segmentation.patchwisemodel import PatchwiseModel
from .graphcuttraining import GraphcutTask


class GraphcutModel(PatchwiseModel):
    def __init__(self, inputsize:int):
        super().__init__(
            module    = GraphcutModule(), 
            inputsize = inputsize,
            patchify  = True,
        )
    
    def start_training(
        self,
        trainsplit: tp.List[tp.Tuple[str,str]],
        valsplit:   tp.List[tp.Tuple[str,str]] | None = None,
        *,
        task_kw:    tp.Dict[str, tp.Any] = {},
        fit_kw:     tp.Dict[str, tp.Any] = {},
    ):
        task_kw = {'inputsize':self.inputsize} | task_kw
        return super()._start_training(
            GraphcutTask, 
            trainsplit, 
            valsplit,
            task_kw = task_kw, 
            fit_kw  = fit_kw
        )


# TODO: 
# - predict a density map which will be used to sample coordinates
# -- could be simply a border map
# -- could be meta-learned automatically by predicting the loss


class GraphcutModuleOutput(tp.NamedTuple):
    raw:         tp.Any
    y_points:    torch.Tensor
    coordinates: torch.Tensor

class GraphcutModule(UNet):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.mha0 = torch.nn.MultiheadAttention(32, num_heads=8, batch_first=True)
        self.mha1 = torch.nn.MultiheadAttention(32, num_heads=8, batch_first=True)
        self.mha2 = torch.nn.MultiheadAttention(32, num_heads=8, batch_first=True)
        self.mha3 = torch.nn.MultiheadAttention(32, num_heads=8, batch_first=True)
        self.prj0 = torch.nn.Conv1d(32, 32, kernel_size=1)
        self.prj1 = torch.nn.Conv1d(32, 32, kernel_size=1)
        self.prj2 = torch.nn.Conv1d(32, 32, kernel_size=1)
        self.prj3 = torch.nn.Conv1d(32, 32, kernel_size=1)
        self.bn0  = torch.nn.BatchNorm1d(32)
        self.bn1  = torch.nn.BatchNorm1d(32)
        self.bn2  = torch.nn.BatchNorm1d(32)
        self.bn3  = torch.nn.BatchNorm1d(32)

    @torch.jit.script_if_tracing
    def forward(self, x:torch.Tensor, n_points:int=512) -> GraphcutModuleOutput:
        y = self._forward_unet(x, return_features=True)

        B,C,H,W = y.shape
        device  = y.device
        coordinates = torch.rand([B, n_points, 1, 2]).to(device) * 2 - 1
        y_points = torch.nn.functional.grid_sample(y, coordinates, mode='bilinear')

        # pass y_points through a transformer
        y_points = y_points[...,0].permute(0,2,1) # B,C,N,1 -> B,N,C

        y_points = torch.cat([y_points[...,:-2], coordinates[:,:,0]], dim=-1)

        y_points, _ = self.mha0(y_points, y_points, y_points)
        y_points    = y_points.permute(0,2,1)
        y_points    = self.prj0(y_points)
        y_points    = self.bn0(y_points)
        y_points    = torch.nn.functional.relu(y_points)
        y_points    = y_points.permute(0,2,1)
        
        y_points, _ = self.mha1(y_points, y_points, y_points)
        y_points    = y_points.permute(0,2,1)
        y_points    = self.prj1(y_points)
        y_points    = self.bn1(y_points)
        y_points    = torch.nn.functional.relu(y_points)
        y_points    = y_points.permute(0,2,1)

        y_points, _ = self.mha2(y_points, y_points, y_points)
        y_points    = y_points.permute(0,2,1)
        y_points    = self.prj2(y_points)
        y_points    = self.bn2(y_points)
        y_points    = torch.nn.functional.relu(y_points)
        y_points    = y_points.permute(0,2,1)

        y_points, _ = self.mha3(y_points, y_points, y_points)
        y_points    = y_points.permute(0,2,1)
        y_points    = self.prj3(y_points)
        y_points    = y_points.permute(0,2,1)

        y_points = y_points.permute(0,2,1)[...,None] # B,N,C -> B,C,N,1

        y_points = normalize(y_points[..., 0])
        

        return GraphcutModuleOutput(
            raw = y,
            y_points = y_points,
            coordinates = coordinates,
        )


def normalize(x:torch.Tensor) -> torch.Tensor:
    '''Normalize to unit length along axis 1'''
    xlength = (x**2).sum(1, keepdims=True)**0.5
    xlength = xlength.clamp_min(1e-5)
    return x/xlength



