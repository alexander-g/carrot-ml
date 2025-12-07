import typing as tp
import warnings; warnings.simplefilter('ignore')
import argparse
import os

import numpy as np
import PIL.Image

from traininglib import datalib, modellib, args, inference



def main(args: argparse.Namespace):
    for inferenceitem in inference.base_inference(args, px_per_mm=args.px_per_mm):
        save_output(inferenceitem, args.output)


def save_output(inferenceitem, destination:str) -> str:
    output    = inferenceitem.output
    inputfile = inferenceitem.inputfile
    outputpath = os.path.join(destination, os.path.basename(inputfile)+'.png')
    if hasattr(output, 'rgb') and output.rgb.dtype == np.uint8:
        PIL.Image.fromarray(output.rgb).save(outputpath)
        return outputpath
    if isinstance(output, np.ndarray):
        if output.dtype in [np.float32, np.float64]:
            output = (output * 255).astype('uint8')
        PIL.Image.fromarray(output).save(outputpath)
        return outputpath
    if 'classmap' in output:
        segmentation_u8 = (output['classmap'] * 255).astype('uint8')
        PIL.Image.fromarray(segmentation_u8).save(outputpath)
        return outputpath
    if 'segmentation' in output:
        segmentation_u8 = (output['segmentation'] * 255).astype('uint8')
        PIL.Image.fromarray(segmentation_u8).save(outputpath)
        return outputpath
    else:
        raise NotImplementedError(type(output))


def get_parser() -> argparse.ArgumentParser:
    parser = args.base_inference_argparser()
    parser.add_argument(
        '--px-per-mm', 
        type = float, 
        help = 'Image resolution',
        required = True, 
    )
    return parser



if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)

    print('Done')
