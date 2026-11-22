import argparse
import hashlib
import json
import os
import typing as tp
import warnings; warnings.simplefilter('ignore')

import numpy as np
import PIL.Image

from traininglib import datalib, modellib, args
from src import util
from src.util import resolve_px_per_mm

import sys
sys.path.insert(0,'cpp/build/')
try:
    import carrot_postprocessing_ext as postp
except ImportError:
    postp = None


def main(args: argparse.Namespace):
    inputs = datalib.collect_inputfiles(args.input)
    nfiles = len(inputs)
    if nfiles == 0:
        print('No inputs')
        return 1
    
    px_per_mm = resolve_px_per_mm(args.px_per_mm, inputs)
    
    versions = {
        'model':         util.get_model_version(args.model),
        'model-weights': util.get_model_weights_version(args.model),
        'dataset':       util.git_version(os.path.dirname(args.input))
    }

    model = modellib.load_model(args.model).to(args.device)
    meta  = util.read_model_metadata(args.model)
    modeltype1 = meta.get('modeltype')

    model2 = None
    modeltype2 = None
    if args.model2 is not None:
        meta2  = util.read_model_metadata(args.model2)
        modeltype2 = meta2.get('modeltype')
        if {modeltype1, modeltype2} != {'carrot-treerings', 'carrot-cells'}:
            print('If two models are provided, they must be cells and treerings')
            return
        
        model2 = modellib.load_model(args.model2).to(args.device)
        versions['model2'] = util.get_model_version(args.model2)
        versions['model2-weights'] = util.get_model_weights_version(args.model2)

        if postp is None:
            print('WARNING: two models provided but postprocessing not available')

    print(versions)

    os.makedirs(args.output, exist_ok=True)

    print(f'Running inference on {nfiles} files.')
    print(f'Using model {args.model}')
    if args.model2 is not None:
        print(f'Second model {args.model2}')
    print(f'Saving output to {args.output}')

    for i, imagefile in enumerate(inputs):
        basename = os.path.basename(imagefile)
        print(f'[{i:4d}/{nfiles}] {basename} {px_per_mm[imagefile]:.1f}', end='\r')
        output  = model.process_image(imagefile, px_per_mm[imagefile])
        outputitem = OutputItem(output, modeltype1)

        outputitem2 = None
        if model2 is not None:
            output2 = model2.process_image(imagefile, px_per_mm[imagefile])
            outputitem2 = OutputItem(output2, modeltype2)

        
        outputpaths = \
            save_output(args.output, InferenceItem(outputitem, outputitem2, imagefile))
        if postp is not None:
            postprocess(outputpaths, imagefile)
    print()

    with open(os.path.join(args.output, 'meta.json'), 'w') as f:
        f.write(json.dumps(versions, indent=2))



class OutputItem(tp.NamedTuple):
    result:    np.ndarray
    modeltype: tp.Literal['carrot-treerings', 'carrot-cells', None]

class InferenceItem(tp.NamedTuple):
    output1:   OutputItem
    output2:   OutputItem|None
    inputfile: str

class OutputPaths(tp.NamedTuple):
    cellmap:     str|None
    treeringmap: str|None

def save_output(destination:str, inferenceitem:InferenceItem) -> OutputPaths:
    output1    = inferenceitem.output1
    output2    = inferenceitem.output2
    inputfile  = inferenceitem.inputfile

    assert output1.result.dtype == np.bool_

    basename   = os.path.basename(inputfile)
    suffix     = '.cells' if output1.modeltype == 'carrot-cells' \
                else '.treerings' if output1.modeltype == 'carrot-treerings' \
                else ''
    outputpath = os.path.join(destination, basename + suffix + '.png')
    PIL.Image.fromarray(output1.result).save(outputpath)

    outputpaths = {}
    outputpaths[output1.modeltype] = outputpath

    if output2 is not None:
        suffix  = '.cells' if output2.modeltype == 'carrot-cells' \
            else '.treerings' if output2.modeltype == 'carrot-treerings' \
            else ''
        outputpath2 = os.path.join(destination, basename + suffix + '.png')
        PIL.Image.fromarray(output2.result).save(outputpath2)

        outputpaths[output2.modeltype] = outputpath2
    
    return OutputPaths(
        cellmap     = outputpaths.get('carrot-cells'),
        treeringmap = outputpaths.get('carrot-treerings'),
    )


def postprocess(outputpaths:OutputPaths, inputfile:str):
    if outputpaths.cellmap is None and outputpaths.treeringmap is None:
        return
    
    imsize  = PIL.Image.open(inputfile).size
    imshape = imsize[::-1]
    output  = postp.postprocess_combined(
        outputpaths.cellmap, 
        outputpaths.treeringmap, 
        workshape = imshape, 
        og_shape  = imshape
    )

    dirname  = os.path.dirname(outputpaths.cellmap or outputpaths.treeringmap) # type: ignore
    basename = os.path.basename(inputfile)
    basepath = os.path.join(dirname, basename)
    if 'ringmap_workshape_png' in output:
        ringmappath = basepath + '.ringmap.png'
        open(ringmappath, 'wb').write( output['ringmap_workshape_png'] )

    if 'ring_points_xy' in output:
        ring_points: list[tuple] = output['ring_points_xy']
        ring_points_path = basepath + '.ring_points.json'
        
        with open(ring_points_path, 'w') as f:
            json.dump([ (a.tolist(), b.tolist()) for a,b in ring_points ], f, indent=2)
    
    if 'cell_info' in output:
        cell_info_path = basepath + '.cell_info.json'
        cell_info: list[dict] = output['cell_info']

        with open(cell_info_path, 'w') as f:
            json.dump( cell_info, f, indent=2 )


def get_parser() -> argparse.ArgumentParser:
    parser = args.base_inference_argparser()
    parser.add_argument(
        '--px-per-mm', 
        help = 'Image resolution, single number or path to a file',
        required = True, 
    )
    parser.add_argument(
        '--model2',
        help = 'Path to the second model, cells or treerings',
        required = False,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    rc = main(args)
    if rc is not None:
        sys.exit(rc)

    print('Done')
