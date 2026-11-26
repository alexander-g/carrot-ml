import argparse
import sys
import time

import torch

from traininglib import args, trainingloop
from src.treerings_yolo import create_dataset_for_yolo, train_yolo_on_treerings
from src.util import verify_reproducibility


def main(args:args.Namespace):
    if not args.allow_dirty:
        errors = verify_reproducibility(args)
        if errors:
            for err in errors:
                print(err)
            sys.exit(1)

    dataset_yaml = create_dataset_for_yolo(
        args.trainsplit, 
        patchsize = args.inputsize, 
        px_per_mm = args.px_per_mm, 
        outputdir = './cache/',
        target_px_per_mm = args.target_px_per_mm,
    )

    pretrained = args.pretrained
    if args.pretrained.lower() == 'none':
        print('No pretrained model provided.')
        pretrained = None
    
    carrotmodel = train_yolo_on_treerings(
        dataset_yaml, 
        epochs           = args.epochs, 
        inputsize        = args.inputsize, 
        weightsfile      = pretrained, 
        verbose          = True,
        splitfile        = args.trainsplit,
        target_px_per_mm = args.target_px_per_mm,
        suffix           = args.suffix,
    )





def get_argparser() -> argparse.ArgumentParser:
    parser = args.base_training_argparser_with_splits(
        default_epochs    = 100,
        default_inputsize = 640,
        # default_lr=1e-4,
    )
    parser.add_argument(
        '--px-per-mm', 
        help     = 'Image resolution, single number or path to a file',
        required = True, 
    )
    parser.add_argument(
        '--target-px-per-mm',
        type     = float,
        help     = 'Input images will be resized to this resolution',
        required = False, 
    )
    parser.add_argument(
        '--pretrained', 
        required = True, # can be "none" but user must explicitly specify
        help     = 'Path to pretrained yolo model as starting point, or "none"'
    )
    parser.add_argument(
        '--allow-dirty',
        action = 'store_true',
        help   = 'Skip reproducibility checks',
    )
    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)
    print('done')
