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
    og_shape:  tp.Tuple[int,int],
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
        classmap_og = resize_tensor(
            classmap[None].float(), 
            og_shape, 
            mode='nearest'
        )[0].to(classmap.dtype)
        instancemap_og = resize_tensor(
            instancemap[None].float(), 
            og_shape, 
            mode='nearest'
        )[0].to(instancemap.dtype)
    else:
        classmap_og = classmap
        instancemap_og = instancemap
    
    instancemap_rgb    = colorize_instancemap(instancemap.numpy())
    instancemap_og_rgb = colorize_instancemap(instancemap_og.numpy())
    return {
        'classmap':    classmap_og.numpy(),
        'instancemap': instancemap_og_rgb,
        'classmap_for_display':    classmap.numpy(),
        'instancemap_for_display': instancemap_rgb,
    }



def load_image(imagepath:str, mode:str='RGB') -> torch.Tensor:
    return torch.as_tensor(
        np.array(
            PIL.Image.open(imagepath).convert(mode)
        )
    )

def resize_tensor(
    x:    torch.Tensor, 
    size: tp.Tuple[int,int], 
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
    


def colorize_instancemap(instancemap:np.ndarray) -> np.ndarray:
    assert instancemap.ndim == 2
    assert instancemap.dtype in [np.int32, np.int64]

    # assuming instance labels are contiguous
    n = instancemap.max()
    hues = np.random.randint(0, 360, size=n)
    sats = np.random.randint(80, 90, size=n)
    vals = np.random.randint(90, 100, size=n)

    colors_hsv = np.concatenate([
        [(0,0,0)],   # black
        np.stack([hues, sats, vals], axis=1)
    ])
    colors_rgb = np.array([ hsv_to_rgb(hsv) for hsv in colors_hsv ])
    instancemap_rgb = colors_rgb[instancemap]
    return instancemap_rgb




def hsv_to_rgb(hsv:tp.Tuple[float,float,float]) -> tp.Tuple[int,int,int]:
    h, s, v = hsv
    # normalize inputs
    h = h % 360
    s = max(0, min(100, s)) / 100.0
    v = max(0, min(100, v)) / 100.0

    if s == 0:
        r = g = b = int(round(v * 255))
        return (r, g, b)

    h_sector = h / 60.0
    i = int(h_sector)  # sector 0..5
    f = h_sector - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    if i == 0:
        r_, g_, b_ = v, t, p
    elif i == 1:
        r_, g_, b_ = q, v, p
    elif i == 2:
        r_, g_, b_ = p, v, t
    elif i == 3:
        r_, g_, b_ = p, q, v
    elif i == 4:
        r_, g_, b_ = t, p, v
    else:  # i == 5
        r_, g_, b_ = v, p, q

    r = int(round(r_ * 255))
    g = int(round(g_ * 255))
    b = int(round(b_ * 255))
    return (r, g, b)


