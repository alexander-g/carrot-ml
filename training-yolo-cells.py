import argparse

import torch

from traininglib import args, trainingloop
from src.cells_yolo import train_yolo_on_cells, HARDCODED_DEFAULT_PATCHSIZE


def main(args:args.Namespace):

    if args.pretrained is None:
        print('No pretrained model provided.')

    train_yolo_on_cells(
        splitfile         = args.trainsplit,
        px_per_mm         = args.px_per_mm,
        inputsize         = args.inputsize,
        batchsize         = args.batchsize,
        epochs            = args.epochs,
        weightsfile       = args.pretrained,
        progress_callback = None,
        reuse_yolo        = args.reuse_yolo,
    )





def get_argparser() -> argparse.ArgumentParser:
    parser = args.base_training_argparser_with_splits(
        default_epochs    = 100,
        default_inputsize = HARDCODED_DEFAULT_PATCHSIZE,
        default_batchsize = 4,
        # default_lr=1e-4,
    )
    parser.add_argument(
        '--px-per-mm', 
        type = float, 
        help = 'Image resolution',
        required = True, 
    )
    parser.add_argument(
        '--reuse-yolo', 
        help = 'Path to previously trained yolo model. Will not train if provided.'
    )
    parser.add_argument('--pretrained', help='Path to pretrained yolo model')
    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)
    print('done')
