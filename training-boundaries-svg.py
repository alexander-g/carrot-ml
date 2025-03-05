import argparse
import typing as tp
import warnings; warnings.simplefilter('ignore')

from traininglib import args
from traininglib.segmentation import (
    start_segmentation_training_from_cli_args,
)
from src.treeringmodel import TreeringDetectionModel


def main(args:argparse.Namespace) -> bool:
    model = TreeringDetectionModel(
        inputsize  = args.inputsize, 
        px_per_mm  = args.px_per_mm,
        patchify   = True,
        #classes             = [Class('ring-boundaries', (255,255,255))]
    )
    return start_segmentation_training_from_cli_args(args, model)



def get_argparser() -> argparse.ArgumentParser:
    parser = args.base_segmentation_training_argparser(
        pos_weight    = 10.0,
        margin_weight = 1.0,
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
    ok   = main(args)
    if ok:
        print('Done')
