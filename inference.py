import argparse
import hashlib
import json
import os
import typing as tp
import warnings; warnings.simplefilter('ignore')

import numpy as np
import PIL.Image

from traininglib import datalib, modellib, args, inference
from src import util



def main(args: argparse.Namespace):
    pairs = datalib.load_file_pairs(args.input)
    if len(pairs) == 0:
        print('No inputs')
        return
    
    inputs = [i for i,_ in pairs]
    px_per_mm = resolve_px_per_mm(args.px_per_mm, inputs)
    
    versions = {
        'model': util.get_model_version(args.model),
        'dataset': util.git_version(os.path.dirname(args.input))
    }
    print(versions)
    
    model  = modellib.load_model(args.model).to(args.device)
    os.makedirs(args.output, exist_ok=True)

    print(f'Running inference on {len(inputs)} files.')
    print(f'Using model {args.model}')
    print(f'Saving output to {args.output}')

    for i, imagefile in enumerate(inputs):
        print(f'[{i:4d}/{len(inputs)}] {os.path.basename(imagefile)} {px_per_mm[imagefile]:.1f}', end='\r')
        output = model.process_image(imagefile, px_per_mm[imagefile])
        save_output(InferenceItem(output, imagefile), args.output)
    print()

    with open(os.path.join(args.output, 'meta.json'), 'w') as f:
        f.write(json.dumps(versions, indent=2))
    

class InferenceItem(tp.NamedTuple):
    output:    tp.Any
    inputfile: str


def save_output(inferenceitem:InferenceItem, destination:str) -> str:
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
    if 'segmentation' in output:
        segmentation_u8 = (output['segmentation'] * 255).astype('uint8')
        PIL.Image.fromarray(segmentation_u8).save(outputpath)
        return outputpath
    else:
        raise NotImplementedError(type(output))

def resolve_px_per_mm(cli_arg:str, inputfiles:list[str]) -> dict[str, float]:
    try:
        px_per_mm = float(cli_arg)
        return {f:px_per_mm for f in inputfiles}
    except ValueError:
        lines = open(cli_arg).read().strip().split('\n')
        basenames = [os.path.basename(f) for f in inputfiles]
        per_file  = {}
        for line in lines:
            filename, px_per_mm_str = line.strip().split(':')
            assert filename in basenames, filename
            filepath = inputfiles[basenames.index(filename)]
            per_file[filepath] = float(px_per_mm_str)
        
        for f in inputfiles:
            assert f in per_file, f
        
        return per_file




def get_parser() -> argparse.ArgumentParser:
    parser = args.base_inference_argparser()
    parser.add_argument(
        '--px-per-mm', 
        help = 'Image resolution, single number or path to a file',
        required = True, 
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)

    print('Done')
