import typing as tp
import warnings; warnings.simplefilter('ignore')
import argparse
import os

import numpy as np
import PIL.Image

from traininglib import datalib, modellib, args


#TODO: code re-use


def inference(args: argparse.Namespace):
    inputs = datalib.collect_inputfiles(args.input)
    model  = modellib.load_model(args.model).to(args.device)
    os.makedirs(args.output, exist_ok=True)
    
    print(f'Running inference on {len(inputs)} files.')
    print(f'Using model {args.model}')
    print(f'Saving output to {args.output}')
    for i, imagefile in enumerate(inputs):
        print(f'[{i:4d}/{len(inputs)}]', end='\r')
        output = model.process_image(imagefile)
        save_output(output.rgb, args.output, imagefile)
    print()


def save_output(output, destination:str, inputfile:str) -> str:
    if isinstance(output, np.ndarray):
        outputpath = os.path.join(destination, os.path.basename(inputfile)+'.png')
        if output.dtype in [np.float32, np.float64]:
            output = (output * 255).astype('uint8')
        PIL.Image.fromarray(output).save(outputpath)
        return outputpath
    else:
        raise NotImplementedError(type(output))



if __name__ == '__main__':
    args = args.base_inference_argparser().parse_args()
    inference(args)

    print('Done')
