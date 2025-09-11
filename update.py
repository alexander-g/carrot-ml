import argparse

import torch

from traininglib import modellib
from src.cc_celldetection import (
    CC_CellsModule, 
    CC_CellsInference, 
    CC_Cells_CARROT,
)
from src.treeringmodel import (
    TreeringsModule, 
    TreeringsInference, 
    Treerings_CARROT,
)


def update(args:argparse.Namespace):
    '''Update a saved inference .torchscript file with new source code'''
    m = modellib.load_model(args.model)
    clsname = m.__class__.__name__

    CARROT_cls:type[CC_Cells_CARROT|Treerings_CARROT]
    inference:CC_CellsInference|TreeringsInference
    if clsname == 'CC_CellsTrainStep':
        module = CC_CellsModule()
        print(module.load_state_dict(m.module.state_dict()))
        inference = CC_CellsInference(module, patchsize=m.inputsize)
        CARROT_cls = CC_Cells_CARROT
    elif clsname == 'TreeringsTrainStep':
        module = TreeringsModule()
        print(module.load_state_dict(m.module.state_dict()))
        inference = TreeringsInference(module, patchsize=m.inputsize)
        CARROT_cls = Treerings_CARROT
    else:
        assert 0, f'Unknown class: {clsname}'
    
    #scripted = torch.jit.script(inference.eval())
    #scripted.save(args.model.replace('.pt.zip', '.torchscript'))

    carrotmodule = CARROT_cls(inference)    # type: ignore
    carrotmodule.save(args.model.replace('.pt.zip', '.carrot.pt.zip'))



def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=update.__doc__)
    parser.add_argument('--model', required=True, help='Path to .pt.zip model')
    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    update(args)
    print('done')
