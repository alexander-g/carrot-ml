import argparse
import typing as tp

from traininglib import args
from traininglib.segmentation import (
    SegmentationModel, 
    Class,
    start_segmentation_training_from_cli_args,
)


def main(args:argparse.Namespace) -> bool:
    model = SegmentationModel(
        inputsize           = args.inputsize, 
        patchify            = True,
        classes             = [Class('ring-boundaries', (0,0,255))]
    )
    return start_segmentation_training_from_cli_args(args, model)


if __name__ == '__main__':
    args = args.base_segmentation_training_argparser(
        pos_weight    = 10.0,
        margin_weight = 1.0,
    ).parse_args()
    ok   = main(args)
    if ok:
        print('Done')
