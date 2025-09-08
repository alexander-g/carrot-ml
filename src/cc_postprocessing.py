import typing as tp

import numpy as np
import PIL.Image
import scipy.ndimage
import torch




def postprocess_cellmapfile(path:str, *a, **kw):
    cellmap = (load_image(path, mode='L') > 127)
    return postprocess_cellmap(cellmap, *a, **kw)


def postprocess_cellmap(
    classmap:  torch.Tensor, 
    og_shape: tp.Tuple[int,int],
    min_object_size_px: int,
):
    assert classmap.ndim  == 2, classmap.shape
    assert classmap.dtype == torch.bool

    # NOTE: scipy is faster than my own implementation
    instancemap_np, _ = \
        scipy.ndimage.label( classmap.numpy(), structure=np.ones([3,3]) )
    instancemap = torch.as_tensor(instancemap_np)
    instancemap = remove_small_objects(
        instancemap, 
        threshold = min_object_size_px,
    )

    if og_shape != instancemap.shape:
        classmap_og = datalib.resize_tensor(
            classmap[None].float(), 
            og_shape, 
            mode='nearest'
        )[0].to(classmap.dtype)
        instancemap_og = datalib.resize_tensor(
            instancemap[None].float(), 
            og_shape, 
            mode='nearest'
        )[0].to(instancemap.dtype)
    else:
        classmap_og = classmap
        instancemap_og = instancemap
    
    return {
        'classmap':    classmap_og.numpy(),
        'instancemap': instancemap_og.numpy(),
        'classmap_for_display':    classmap.numpy(),
        'instancemap_for_display': instancemap.numpy(),
    }



def load_image(imagepath:str, mode:str='RGB') -> torch.Tensor:
    return torch.as_tensor(
        np.array(
            PIL.Image.open(imagepath).convert(mode)
        )
    )

def resize_tensor(
    x:    torch.Tensor, 
    size: tp.List[int], 
    mode: str,
    align_corners: tp.Optional[bool] = None,
) -> torch.Tensor:
    assert len(x.shape) in [3,4]

    x0 = x
    if len(x0.shape) == 3:
        x = x[np.newaxis]
    y = torch.nn.functional.interpolate(x, size, mode=mode, align_corners=align_corners)
    if len(x0.shape) == 3:
        y = y[0]
    return y


def remove_small_objects(instancemap:torch.Tensor, threshold:int) -> torch.Tensor:
    assert instancemap.ndim == 2
    assert instancemap.dtype == torch.int64 or instancemap.dtype == torch.int32
    
    labels, counts = torch.unique(instancemap, return_counts=True)
    good_labels = labels[counts > threshold]

    mask = torch.isin(instancemap, good_labels) & (instancemap != 0)
    return mask * instancemap


