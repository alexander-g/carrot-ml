import argparse
import typing as tp
import warnings; warnings.simplefilter('ignore')

from traininglib import args, datalib
from traininglib.modellib import (
    start_training_from_cli_args,
)
from src.cellsmodel import CellDetectionModel


def main(args:argparse.Namespace) -> bool:
    model = CellDetectionModel(
        px_per_mm = args.px_per_mm,
    )
    trainsplit = datalib.load_file_pairs(args.trainsplit)
    return start_training_from_cli_args(args, model, trainsplit)



def get_argparser() -> argparse.ArgumentParser:
    parser = args.base_training_argparser_with_splits(default_lr=1e-4)
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
