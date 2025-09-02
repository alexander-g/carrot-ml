import argparse
import glob
import json
import os
import typing as tp

import tifffile
import numpy as np
import torch

from traininglib import datalib
from traininglib.segmentation import connectedcomponents as concom
from src.evaluation import IoU_matrix, best_iou_matches_greedy



def main(args:argparse.Namespace):
    file_pairs = collect_output_annotation_pairs(
        args.outputs, 
        args.annotations, 
        '.jpg.instances.tiff'
    )
    all_metrics = []
    for i, (annotationfile, outputfile) in enumerate(file_pairs):
        print(f'[{i:3d}/{len(file_pairs)}] {os.path.basename(annotationfile)}')
        all_metrics += [evaluate_file_pair(outputfile, annotationfile)]
    outputfile = os.path.join(args.outputs, 'metrics.json')
    metrics = save_metrics(all_metrics, [a for a,_ in file_pairs], outputfile)
    # print('merged metrics:', metrics)


def collect_output_annotation_pairs(
    outputfolder:   str, 
    annotationfile: str,
    file_ending:    str,
) -> tp.List[tp.Tuple[str,str]]:
    pattern = os.path.join(outputfolder, f'*{file_ending}')
    outputfiles = sorted(glob.glob( pattern ))
    input_annotation_pairs = datalib.load_file_tuples(annotationfile, ',', 2)
    annotationfiles = [af for _,af in input_annotation_pairs]

    output_annotation_pairs = {
        af:of for of in outputfiles for af in annotationfiles 
            if are_files_matching(of, af, file_ending, '.png')
    }
    
    missing_outputfiles = [
        os.path.basename(of) for of in outputfiles 
            if of not in output_annotation_pairs.values()
    ]
    missing_annotationfiles = [
        os.path.basename(af) for af in annotationfiles 
            if af not in output_annotation_pairs.keys()
    ]
    if len(missing_outputfiles):
        print('Unmatched outputfiles:',     missing_outputfiles)
    if len(missing_annotationfiles):
        print('Unmatched annotationfiles:', missing_annotationfiles)
    return list(output_annotation_pairs.items())


def are_files_matching(
    outputfile:str, 
    annotationfile:str,
    output_file_ending:str,
    annotation_file_ending:str,
) -> bool:
    outputfile = \
        os.path.basename(outputfile).replace(output_file_ending, '')
    annotationfile = \
        os.path.basename(annotationfile).replace(annotation_file_ending, '')
    return outputfile == annotationfile[:len(outputfile)]


def evaluate_file_pair(outputfile:str, annotationfile:str):
    output = torch.as_tensor(tifffile.imread(outputfile)).to(torch.int32)
    annotation = datalib.load_image(annotationfile, mode='L')
    annotation = concom.connected_components_patchwise(
        annotation[None] > 0.5, 
        patchsize=512
    )[0,0]
    annotation = torch.unique(annotation, return_inverse=True)[1].to(torch.int32)

    iou = IoU_matrix(output, annotation, zero_out_zero=True)
    matches, best_ious = best_iou_matches_greedy(iou, threshold=0.7)
    
    TP = len(matches)
    FP = int(output.max() - TP)
    FN = int(annotation.max() - TP)

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'precision': TP / (TP+FP),
        'recall':    TP / (TP+FN),
    }

def save_metrics(
    metrics: tp.List[tp.Dict[str,float]], 
    files:   tp.List[str], 
    outputfile: str,
):
    jsondata = merge_metrics(metrics)
    for m,f in zip(metrics, files):
        jsondata[f] = m
    open(outputfile, 'w').write(json.dumps(jsondata, indent=2))
    return jsondata['mean']


def merge_metrics(metrics:tp.List[tp.Dict[str,float]]) -> tp.Dict:
    if len(metrics) == 0:
        return {}
    
    meankeys = ['precision', 'recall']
    sumkeys  = ['TP', 'FP', 'FN']
    meanmetrics = {k:float(np.mean( [m[k]  for m in metrics] )) for k in meankeys}
    summetrics  = {k:int(np.sum( [m[k]  for m in metrics] )) for k in sumkeys}
    summetrics  = summetrics | { # type: ignore
        'precision': summetrics['TP'] / max(summetrics['TP'] + summetrics['FP'], 1),
        'recall': summetrics['TP'] / max(summetrics['TP'] + summetrics['FN'], 1),
    }
    return {
        'mean': meanmetrics,
        'sum':  summetrics,
    }


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', help='Path to outputs')
    parser.add_argument('--annotations', help='Path to split file')
    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)
    print('done')

