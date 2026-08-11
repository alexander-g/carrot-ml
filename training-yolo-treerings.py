import argparse
import time

import torch

from traininglib import args, trainingloop
from src.treerings_yolo import create_dataset_for_yolo, train_yolo_on_treerings


def main(args:args.Namespace):
    dataset_yaml = create_dataset_for_yolo(
        args.trainsplit, 
        patchsize = args.inputsize, 
        px_per_mm = args.px_per_mm, 
        outputdir = './cache/',
    )

    if args.pretrained is None:
        print('No pretrained model provided.')
    carrotmodel = train_yolo_on_treerings(dataset_yaml, epochs=args.epochs, inputsize=args.inputsize, weightsfile=args.pretrained, verbose=True)
    #carrotmodel.save()

    breakpoint()





def get_argparser() -> argparse.ArgumentParser:
    parser = args.base_training_argparser_with_splits(
        default_epochs    = 100,
        default_inputsize = 640,
        # default_lr=1e-4,
    )
    parser.add_argument(
        '--px-per-mm', 
        type = float, 
        help = 'Image resolution',
        required = True, 
    )
    parser.add_argument('--pretrained', help='Path to pretrained yolo model')
    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)
    print('done')
