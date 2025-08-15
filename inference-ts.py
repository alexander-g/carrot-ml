import argparse
import json
import os
import typing as tp

import matplotlib.cm as mplcm
import numpy as np
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None
import tifffile
import torch

from traininglib import datalib, args


def main(args:argparse.Namespace):
    model = torch.jit.load(args.model)
    if torch.cuda.is_available():
        model = model.cuda()

    imagepaths = datalib.collect_inputfiles(args.input)
    for i, imagepath in enumerate(imagepaths):
        print(f'[{i:3d}/{len(imagepaths)}]', end='\r')
        x = load_image(imagepath)

        with torch.no_grad():
            output = model(x, batchsize=4)
        
        outputdir = args.output
        if outputdir == './inference':
            outputdir = os.path.join(outputdir, os.path.basename(args.model))
        save_output(
            outputdir,
            imagepath,
            output
        )
    print()


def save_output(
    outputdir: str, 
    imagepath: str,
    output:    tp.Dict[str, torch.Tensor]
):
    os.makedirs(outputdir, exist_ok=True)

    outputname = f'{os.path.basename(imagepath)}'
    outputpath_raw = os.path.join(outputdir, 'raw', outputname)+'.raw.png'
    os.makedirs(os.path.dirname(outputpath_raw), exist_ok=True)
    raw = output['raw'].float()
    datalib.write_image_tensor(outputpath_raw, raw)

    outputpath_instances = \
        os.path.join(outputdir, 'instances', outputname)+'.instances.tiff'
    os.makedirs(os.path.dirname(outputpath_instances), exist_ok=True)
    imap = output['instancemap']
    imap = randomize_instancemap(imap)
    datalib.write_image_tensor(outputpath_instances, imap.to(torch.int32))

    outputpath_vis = os.path.join(outputdir, 'vis', outputname+'.vis.png')
    os.makedirs(os.path.dirname(outputpath_vis), exist_ok=True)
    imap_vis = mplcm.gist_ncar( imap / imap.max() ) # type: ignore
    imap_vis = torch.as_tensor( imap_vis )[..., :3].permute(2,0,1)
    imap_vis = imap_vis * (imap != 0)
    datalib.write_image_tensor(outputpath_vis, imap_vis)


def randomize_instancemap(imap:torch.Tensor) -> torch.Tensor:
    assert imap.dtype == torch.int64
    mask = (imap != 0)
    # make sure instance values are sequential
    imap = torch.unique(imap, return_inverse=True)[1]
    
    randomvalues = torch.randperm( int(imap.max()) )
    randomvalues = torch.cat([torch.zeros_like(randomvalues[:1]), randomvalues])
    random_imap  = randomvalues[imap]
    random_imap  = random_imap * mask
    valdiff = torch.diff( torch.unique(random_imap) )
    #breakpoint()
    assert (valdiff == 1).all()
    return random_imap




def load_image(path:str) -> torch.Tensor:
    im = PIL.Image.open(path).convert('RGB')
    return torch.as_tensor(np.array(im))


def get_parser() -> argparse.ArgumentParser:
    return args.base_inference_argparser()


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
    print('done')

