import typing as tp

import numpy as np
import torch
import torchvision

from traininglib import datalib
from traininglib.trainingtask import TrainingTask, Loss, Metrics
from traininglib.segmentation import margin_loss_fn, SegmentationDataset



class GraphcutTask(TrainingTask):
    def __init__(self, *a, inputsize:int, **kw):
        super().__init__(*a, **kw)
        self.inputsize   = inputsize
        self.cropfactors = (0.75, 1.33)
        self.patchify    = True
        self.rotate      = True

    @torch.jit.script_if_tracing
    def training_step(self, raw_batch) -> tp.Tuple[Loss, Metrics]:
        x, t = self.prepare_batch(raw_batch)
        output = self.basemodule(x)

        y_points = output.y_points
        t_points = \
            torch.nn.functional.grid_sample(t.float(), output.coordinates, mode='nearest')
        t_points = t_points[...,0].long()
        
        loss = crosswise_similarity_loss(y_points, t_points)
        #dmat = distance_matrix(output.coordinates[...,0,:])
        #assert loss.shape == dmat.shape
        #loss = loss[dmat < 0.2]
        loss = loss.mean()

        logs = {'xsim':float(loss)}
        return loss, logs

    #TODO: code re-use
    def create_dataloaders(
        self, 
        trainsplit: tp.List, 
        valsplit:   tp.List|None = None, 
        **ld_kw,
    ) -> tp.Tuple[tp.Iterable, tp.Iterable|None]:
        patchsize_train = patchsize_val = None
        if self.patchify:
            # NOTE: *2 for cropping during augmentation
            patchsize_train = self.inputsize * 2
            patchsize_val   = self.inputsize
        
        ds_train = SegmentationDataset(trainsplit, patchsize_train)
        ld_train = datalib.create_dataloader(ds_train, shuffle=True, **ld_kw)
        ld_val   = None
        if valsplit is not None:
            ds_val = SegmentationDataset(valsplit, patchsize_val)
            ld_val = datalib.create_dataloader(ds_val, shuffle=False, **ld_kw)
        return ld_train, ld_val

    # TODO: code re-use!!
    def augment(self, batch) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        x,t = batch
        new_x: tp.List[torch.Tensor] = []
        new_t: tp.List[torch.Tensor] = []
        for xi,ti in zip(x,t):
            if self.cropfactors is not None:
                xi,ti = datalib.random_crop(
                    xi, 
                    ti, 
                    patchsize   = self.inputsize, 
                    modes       = ['bilinear', 'nearest'], 
                    cropfactors = self.cropfactors
                )
            xi,ti = datalib.random_rotate_flip(xi, ti, rotate=self.rotate)
            new_x.append(xi)
            new_t.append(ti)
        x = torch.stack(new_x)
        t = torch.stack(new_t)
        return x,t

    def prepare_batch(self, raw_batch) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        x,t = raw_batch
        assert t.dtype == torch.uint8
        assert t.ndim  == 4 and t.shape[1] == 3, t.shape
        t   = t[:,:1]

        assert x.dtype == torch.float32

        x,t = datalib.to_device(x, t, device=self.device)
        x,t = self.augment((x,t))
        return x,t

def crosswise_similarity_loss(y:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
    # y: [B,C,n] float32
    # t: [B,1,n] int64
    assert y.ndim == 3, y.shape
    assert t.ndim == 3 and t.shape[1] == 1
    assert t.dtype == torch.long
    assert y.shape[-1] == t.shape[-1]

    similarity_matrix = torch.einsum('bcn,bcm->bnm', y, y)
    t = t[:,0]
    target_matrix = (t[:,:,None] == t[:,None])
    assert similarity_matrix.shape == target_matrix.shape

    return torch.nn.functional.binary_cross_entropy(
        (similarity_matrix/2+0.5).clamp(0,1), 
        target_matrix.float(), 
        reduction='none'
    )

    target_matrix = target_matrix.float() * 2 -1   # -1..+1

    #return torch.nn.functional.l1_loss(similarity_matrix, target_matrix, reduction='none')
    #return torch.nn.functional.mse_loss(similarity_matrix, target_matrix)

def distance_matrix(coordinates:torch.Tensor) -> torch.Tensor:
    assert coordinates.ndim == 3 and coordinates.shape[-1] == 2
    return ((coordinates[...,None,:,:] - coordinates[...,None,:])**2).sum(-1)**0.5
