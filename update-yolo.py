import argparse
import os

import torch
import ultralytics

from traininglib import modellib
from src.cells_yolo import CellsYOLO_CARROT, CellsYOLO_Module
from src.treerings_yolo import TreeringsYOLO_CARROT, TreeringsYOLO_Module, TreeringsInference




def update(args:argparse.Namespace):
    '''Update a saved yolo file with new source code'''
    CARROT_cls:type[CellsYOLO_CARROT|TreeringsYOLO_CARROT]
    inference: CellsYOLO_Module|TreeringsInference


    if args.model.endswith('.pt'):
        m = ultralytics.YOLO(args.model)  # type: ignore[attr-defined]
        assert args.px_per_mm is not None

        if m.args['task'] == 'segment':
            CARROT_cls = CellsYOLO_CARROT
            inference  = CellsYOLO_Module(m, args.px_per_mm)
        elif m.args['task'] == 'semantic':
            CARROT_cls = TreeringsYOLO_CARROT
            inference  = TreeringsInference(
                TreeringsYOLO_Module(m, args.px_per_mm),
                patchsize = m.args['imgsz'],
            )
        else:
            print(f'Unknown yolo model: {m.args["task"]}')
            return
    elif args.model.endswith('.pt.zip'):
        m = modellib.load_model(args.model)
        clsname = m.__class__.__name__
        assert 0, NotImplemented
    else:
        print(f'Unknown file type: {args.model}')
        return
    
    
    #scripted = torch.jit.script(inference.eval())
    #scripted.save(args.model.replace('.pt.zip', '.torchscript'))

    carrotmodule = CARROT_cls(inference)    # type: ignore

    filename  = os.path.splitext(os.path.basename(args.model))[0] + '.carrot.pt.zip'
    outputdir = args.outputdir or os.path.dirname(args.model)
    os.makedirs(outputdir, exist_ok=True)
    outputpath = os.path.join(outputdir, filename)
    print(f'Saving to {outputpath}')
        
    carrotmodule.save(outputpath)



def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=update.__doc__)
    parser.add_argument('--model', required=True, help='Path to a yolo .pt model')
    parser.add_argument('--px-per-mm', type=float)
    parser.add_argument('--outputdir', type=str)
    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    update(args)
    print('done')
