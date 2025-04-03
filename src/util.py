import tifffile

import torch

from traininglib import datalib



def load_and_scale_image(path:str, scale:float = 1.0) -> torch.Tensor:
    '''Load and scale an image, if a tiff, patchwise to reduce memory'''
    
    if path.endswith('.tiff') or path.endswith('.tif'):
        return load_and_scale_tiff(path, scale)
    else:
        x   = datalib.load_image(path, to_tensor=True, normalize=False)
        H,W = x.shape[-2:]
        newshape = [ int(H * scale), int(W * scale) ]
        x = datalib.resize_tensor(x, newshape, 'bilinear')
        return x


def load_and_scale_tiff(path:str, scale:float, patchsize:int=10240) -> torch.Tensor:
    '''Load and scale a tiff image, patchwise to reduce memory'''
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        og_shape = page.shape
        H,W,C = og_shape

    newshape = [ C, int(H * scale), int(W * scale) ]
    result = torch.zeros(newshape, dtype=torch.uint8)

    try:
        imdata = tifffile.memmap(path, mode='r')
    except ValueError:
        # cannot memmap if compressed
        with tifffile.TiffFile(path) as tif:
            imdata = tif.pages[0].asarray()  # type: ignore

    for i in range(0, H, patchsize):
        for j in range(0, W, patchsize):
            patch = torch.as_tensor(
                imdata[i:i+patchsize, j:j+patchsize]
            ).permute(2,0,1)
            h,w = patch.shape[-2:]
            newpatchshape = [ int(h * scale), int(w * scale) ]
            patch = datalib.resize_tensor(patch, newpatchshape, 'bilinear')
            i_ = int(i*scale)
            j_ = int(j*scale)
            result[:, i_:i_+patch.shape[-2], j_:j_+patch.shape[-1]] = patch
    return result



