import argparse
import os
import typing as tp

import torch
import ultralytics

from traininglib import modellib
from src.cells_yolo import (
    CellsYOLO_CARROT, 
    CellsYOLO_Module,
    MaskHead,
    create_new_yolo_det_model,
    export_cells_yolo_to_onnx
)
from src.cells_yolo_maskhead import infer_maskhead_kw
from src.treerings_yolo import (
    TreeringsYOLO_CARROT, 
    TreeringsYOLO_Module, 
    TreeringsInference, 
    create_new_yolo_sem_model,
    export_treerings_yolo_to_onnx,
)




def update(args:argparse.Namespace):
    '''Update a saved yolo file with new source code'''
    CARROT_cls:type[CellsYOLO_CARROT|TreeringsYOLO_CARROT]
    inference: CellsYOLO_Module|TreeringsInference

    outputdir = os.path.dirname(args.model)
    export_onnx_fn: tp.Callable|None = None

    if args.model.endswith('.pt'):
        m = ultralytics.YOLO(args.model)  # type: ignore[attr-defined]
        assert args.px_per_mm is not None, '--px-per-mm required for raw YOLO files'

        if m.args['task'] == 'detect':
            assert args.maskhead, '--maskhead required for raw YOLO detect files'
            maskhead = recreate_maskhead_from_carrotptzip(args.maskhead)
            CARROT_cls = CellsYOLO_CARROT
            inference  = CellsYOLO_Module(m, maskhead, args.px_per_mm)
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
        if clsname == 'TreeringsYOLO_CARROT':
            yolomodel = create_new_yolo_sem_model()
            px_per_mm = m.module.module.px_per_mm
            patchsize = m.module.patchsize
            inference = TreeringsInference(
                TreeringsYOLO_Module(yolomodel, px_per_mm),
                patchsize = patchsize,
            ).eval()
            status = inference.load_state_dict(m.module.state_dict())
            print(status)
            CARROT_cls = TreeringsYOLO_CARROT

            basename = os.path.basename(args.model).removesuffix('.pt.zip')
            outputpath_onnx = os.path.join(outputdir, basename+'.onnx')
            export_onnx_fn = export_treerings_yolo_to_onnx

        elif clsname == 'CellsYOLO_CARROT':
            yolomodel   = create_new_yolo_det_model('/tmp', inputsize=800)
            px_per_mm   = m.module.px_per_mm
            patchsize   = m.module.inputsize
            maskhead    = recreate_maskhead_from_carrotptzip(args.model)
            inference   = CellsYOLO_Module(yolomodel, maskhead, px_per_mm).eval()
            status      = inference.load_state_dict(m.module.state_dict())
            print(status)
            CARROT_cls = CellsYOLO_CARROT

            basename = os.path.basename(args.model).removesuffix('.pt.zip')
            outputpath_onnx = os.path.join(outputdir, basename+'.onnx')
            export_onnx_fn = export_cells_yolo_to_onnx
            # export_cells_yolo_to_onnx(inference, outputpath_onnx, patchsize)
        else:
            print(f'Unknown model class: {clsname}')
            return
    else:
        print(f'Unknown file type: {args.model}')
        return
    
    carrotmodule = CARROT_cls(inference)    # type: ignore

    basename   = os.path.basename(args.model).removesuffix('.pt.zip')
    outputpath = os.path.join(outputdir, basename+'.carrot.pt.zip')
    print()
    print(f'Saving to {outputpath}')
    carrotmodule.save(outputpath)

    if export_onnx_fn is not None:
        outputpath_onnx = os.path.join(outputdir, basename+'.onnx')
        print(f'Saving ONNX to {outputpath_onnx}')
        export_onnx_fn(carrotmodule.module, outputpath_onnx, patchsize)


def recreate_maskhead_from_carrotptzip(path:str):
    old_maskhead = modellib.load_model(path).module.maskhead
    maskhead_kw = infer_maskhead_kw(old_maskhead)
    maskhead    = MaskHead(**maskhead_kw)
    status      = maskhead.load_state_dict(old_maskhead.state_dict())
    return maskhead


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=update.__doc__)
    parser.add_argument('--model', required=True, help='Path to a yolo .pt model')
    parser.add_argument('--px-per-mm', type=float)
    parser.add_argument('--maskhead', help='Path to .carrot.pt.zip')
    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    update(args)
    print('done')
