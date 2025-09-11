import argparse

import torch

from traininglib import args, trainingloop
from src.treeringmodel import (
    TreeringsModule,
    TreeringsTrainStep,
    TreeringsDataset,
)


def main(args:args.Namespace):
    module  = TreeringsModule()
    step    = TreeringsTrainStep(module, inputsize=args.inputsize)
    # NOTE: *2 because of cropping augmentations
    dataset = TreeringsDataset(
        args.trainsplit, 
        patchsize = args.inputsize*2, 
        px_per_mm = args.px_per_mm
    )

    # NOTE: threaded because getting errors otherwise
    ld_kw = {'loader_type': 'threaded'}
    step, paths = \
        trainingloop.start_training_from_cli_args(args, step, dataset, ld_kw)
    
    #inference = CC_CellsInference(step.module, patchsize=args.inputsize)
    #torch.jit.script(inference).save(paths.modelpath+'.torchscript')





def get_argparser() -> argparse.ArgumentParser:
    parser = args.base_training_argparser_with_splits(
        default_epochs=1000,
    )
    parser.add_argument(
        '--px-per-mm', 
        type = float, 
        help = 'Image resolution',
        required = True, 
    )
    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)
    print('done')
