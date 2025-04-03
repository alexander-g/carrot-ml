import argparse
import os

from traininglib import modellib
from src.treeringmodel import TreeringDetectionModel, HARDCODED_GOOD_RESOLUTION


def main(args:argparse.Namespace) -> bool:
    old_model = modellib.load_model(args.model)
    if hasattr(old_model, 'px_per_mm'):
        px_per_mm = old_model.px_per_mm
    elif hasattr(old_model, 'scale'):
        px_per_mm = HARDCODED_GOOD_RESOLUTION/old_model.scale
    else:
        px_per_mm = 1.0
    new_model = TreeringDetectionModel(
        inputsize = getattr(old_model, 'inputsize', 512),
        px_per_mm = px_per_mm,
    )
    new_model = new_model.eval().requires_grad_(False)
    print(new_model.load_state_dict(old_model.state_dict()))

    new_modelpath = args.model.replace('.pt.zip', '.update.pt.zip')
    print('Saving updated model to: ', new_modelpath)
    new_model.save(new_modelpath)
    return True



def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to model that should be updated')
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok = main(args)
    if ok:
        print('Done')
