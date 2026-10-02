import argparse

import torch

from traininglib import args, trainingloop
from src.cells_yolo_maskhead import (
    MaskHead, 
    MaskHeadTrainStep,
    InstanceDataset,
)


def main(args:args.Namespace):
    module = MaskHead()
    step   = MaskHeadTrainStep(module)
    dataset = InstanceDataset.from_splitfile(
        args.trainsplit, 
        patchsize = args.inputsize, 
        px_per_mm = args.px_per_mm
    )

    # NOTE: threaded because getting errors otherwise
    ld_kw = {'loader_type': 'threaded'}
    step, paths = \
        trainingloop.start_training_from_cli_args(args, step, dataset, ld_kw=ld_kw)
    





def get_argparser() -> argparse.ArgumentParser:
    parser = args.base_training_argparser_with_splits(
        default_epochs=100,
        default_inputsize=640,
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
