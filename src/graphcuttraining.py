import typing as tp

import numpy as np
import torch
import torchvision

from traininglib import datalib
from traininglib.trainingtask import Loss, Metrics
from traininglib.segmentation import margin_loss_fn, PatchwiseTrainingTask



class GraphcutTask(PatchwiseTrainingTask):

    @torch.jit.script_if_tracing
    def training_step(self, raw_batch) -> tp.Tuple[Loss, Metrics]:
        x, t = self.prepare_batch(raw_batch, augment=True)
        t    = t[:,:1]
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
        # NOTE: *0.9999 to because values might overshoot above 1 slightly
        # clamping caused issues
        (similarity_matrix*0.99999/2+0.5), 
        target_matrix.float(), 
        reduction='none'
    )

    target_matrix = target_matrix.float() * 2 -1   # -1..+1

    #return torch.nn.functional.l1_loss(similarity_matrix, target_matrix, reduction='none')
    #return torch.nn.functional.mse_loss(similarity_matrix, target_matrix)

def distance_matrix(coordinates:torch.Tensor) -> torch.Tensor:
    assert coordinates.ndim == 3 and coordinates.shape[-1] == 2
    return ((coordinates[...,None,:,:] - coordinates[...,None,:])**2).sum(-1)**0.5
