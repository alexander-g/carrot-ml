import argparse

import torch

from traininglib import args, trainingloop
from src.cc_celldetection import (
    CC_CellsModule, 
    CC_CellsTrainStep,
    CC_CellsInference,
    CC_CellsDataset,
)


def main(args:args.Namespace):
    module  = CC_CellsModule()
    step    = CC_CellsTrainStep(module, inputsize=args.inputsize)
    # NOTE: *2 because of cropping augmentations
    dataset = CC_CellsDataset(args.trainsplit)

    # NOTE: threaded because getting errors otherwise
    ld_kw = {'loader_type': 'threaded'}
    step, paths = \
        trainingloop.start_training_from_cli_args(args, step, dataset, ld_kw)
    
    inference = CC_CellsInference(step.module, patchsize=args.inputsize)
    torch.jit.script(inference).save(paths.modelpath+'.torchscript')





def get_argparser() -> argparse.ArgumentParser:
    parser = args.base_training_argparser_with_splits(
        default_epochs=1000,
    )
    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)
    print('done')
