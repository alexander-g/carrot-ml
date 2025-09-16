import argparse

import torch

from traininglib import args, trainingloop
from src.maskrcnn_celldetection import (
    MaskRCNN_CellsModule, 
    MaskRCNN_TrainStep,
    InstanceDataset,
)


def main(args:args.Namespace):
    module  = MaskRCNN_CellsModule(inputsize=args.inputsize)
    step    = MaskRCNN_TrainStep(module, inputsize=args.inputsize)
    # NOTE: *2 because of cropping augmentations
    dataset = InstanceDataset(
        args.trainsplit, 
        patchsize = args.inputsize*2, 
        px_per_mm = args.px_per_mm
    )

    # NOTE: threaded because getting errors otherwise
    ld_kw = {'loader_type': 'threaded'}
    step, paths = \
        trainingloop.start_training_from_cli_args(args, step, dataset, ld_kw)
    
    #inference = TODO(step.module, patchsize=args.inputsize)
    #torch.jit.script(inference).save(paths.modelpath+'.torchscript')





def get_argparser() -> argparse.ArgumentParser:
    parser = args.base_training_argparser_with_splits(
        default_epochs=100,
        default_inputsize=800,
        default_lr=1e-4,
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
