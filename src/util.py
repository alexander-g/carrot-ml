import typing as tp

import numpy as np
import PIL.Image
import tifffile
import torch


# first height, then width
Shape = tp.Tuple[int,int]
ImageAndOGShape = tp.Tuple[torch.Tensor, Shape]


# NOTE: re-implementing here, to avoid traininglib requirement in carrot
def load_image(imagepath:str, mode:str='RGB') -> torch.Tensor:
    return torch.as_tensor(
        np.array(
            PIL.Image.open(imagepath).convert(mode)
        )
    )

def resize_tensor(
    x:     torch.Tensor, 
    shape: Shape, 
    mode:  str,
    align_corners: tp.Optional[bool] = None,
) -> torch.Tensor:
    assert len(x.shape) in [3,4]

    x0 = x
    if len(x0.shape) == 3:
        x = x[np.newaxis]
    y = torch.nn.functional.interpolate(x, shape, mode=mode, align_corners=align_corners)
    if len(x0.shape) == 3:
        y = y[0]
    return y


def load_and_scale_image(path:str, scale:float = 1.0) -> ImageAndOGShape:
    '''Load and scale an image, if a tiff, patchwise to reduce memory'''
    
    if path.endswith('.tiff') or path.endswith('.tif'):
        return load_and_scale_tiff(path, scale)
    else:
        x   = load_image(path)
        x   = x.permute(2,0,1)
        H,W = x.shape[-2:]
        newshape = ( int(H * scale), int(W * scale) )
        x = resize_tensor(x, newshape, 'bilinear')
        x = x.permute(1,2,0)
        return x, (H,W)


def load_and_scale_tiff(path:str, scale:float, patchsize:int=10240) -> ImageAndOGShape:
    '''Load and scale a tiff image, patchwise to reduce memory'''
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        og_shape = page.shape
    if len(og_shape) == 3:
        H,W,C = og_shape
    elif len(og_shape) == 2:
        H,W = og_shape
    else:
        # ???? should not happen
        raise RuntimeError(f'Unexpected image shape: {og_shape}')
    C = 3 
    newshape = [ C, int(H * scale), int(W * scale) ]
    result = torch.zeros(newshape, dtype=torch.uint8)

    try:
        imdata = tifffile.memmap(path, mode='r')
    except ValueError:
        # cannot memmap if compressed
        with tifffile.TiffFile(path) as tif:
            imdata = tif.pages[0].asarray()  # type: ignore

    if len(imdata.shape) == 2:
        imdata = imdata[..., None] # type: ignore

    for i in range(0, H, patchsize):
        for j in range(0, W, patchsize):
            patch = torch.as_tensor(
                imdata[i:i+patchsize, j:j+patchsize]
            ).permute(2,0,1)
            h,w = patch.shape[-2:]
            newpatchshape = ( int(h * scale), int(w * scale) )
            patch = resize_tensor(patch, newpatchshape, 'bilinear')
            i_ = int(i*scale)
            j_ = int(j*scale)
            result[:, i_:i_+patch.shape[-2], j_:j_+patch.shape[-1]] = patch[:3]
    result = result.permute(1,2,0)
    return result, (H,W)


def scale_points_xy(points_xy:np.ndarray, from_shape:Shape, to_shape:Shape) -> np.ndarray:
    '''Points in XY format, shapes in width-height format'''
    assert points_xy.ndim == 2 and points_xy.shape[-1] == 2

    scale = np.array([
        to_shape[1] / from_shape[1],
        to_shape[0] / from_shape[0],
    ])
    return points_xy * scale


def scale_points_yx(points_yx:np.ndarray, from_shape:Shape, to_shape:Shape) -> np.ndarray:
    '''Points in YX format, shapes in width-height format'''
    assert points_yx.ndim == 2 and points_yx.shape[-1] == 2
    points_xy = points_yx[:,::-1]
    scaled_xy = scale_points_xy( points_xy, from_shape, to_shape )
    scaled_yx = scaled_xy[:,::-1]
    return scaled_yx