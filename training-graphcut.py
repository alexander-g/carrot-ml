import argparse
import typing as tp

from traininglib import args, datalib
from traininglib.modellib import start_training_from_cli_args
from src.graphcutmodel import GraphcutModel


def main(args:argparse.Namespace) -> bool:
    model = GraphcutModel(inputsize = args.inputsize)
    trainsplit = datalib.load_file_pairs(args.trainsplit)
    return start_training_from_cli_args(args, model, trainsplit)


if __name__ == '__main__':
    args = args.base_training_argparser_with_splits().parse_args()
    ok   = main(args)
    if ok:
        print('Done')


# - inference:
# -- patchify
# - training:
# -- 
