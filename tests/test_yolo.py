import os
import tempfile

from src.cells_yolo import (
    CellsYOLO_Module, 
    CellsYOLO_Module, 
    MaskRCNN_Cells_CARROT,
    box_overlap_matrix,
    box_overlap_filter,
)
from src.cells_yolo_maskhead import MaskHead
from src.treerings_yolo import TreeringsYOLO_Module, Treerings_CARROT, TreeringsInference
from src.maskrcnn_celldetection import (
    stitch_and_relabel_instancemaps_from_grid,
    split_overlapping_masks,
)

import numpy as np
import ultralytics
import PIL.Image
import torch



def test_treerings_inference():
    m = ultralytics.YOLO('vendor/ultralytics/ultralytics/cfg/models/26/yolo26-sem.yaml')
    inputsize = m.args['imgsz']
    module = TreeringsYOLO_Module(m, px_per_mm=250).eval()
    model  = Treerings_CARROT(TreeringsInference(module, patchsize=inputsize))

    tempdir = tempfile.TemporaryDirectory()


    # basic, same size as yolo expects
    imagedata = (np.random.random([inputsize, inputsize, 3]) * 255).astype('uint8')
    imagepath = os.path.join(tempdir.name, 'image0.png')
    PIL.Image.fromarray(imagedata).save(imagepath)

    model.process_image(imagepath, px_per_mm=250)



    # bug: odd image size
    imagedata = (np.random.random([489, inputsize, 3]) * 255).astype('uint8')
    imagepath = os.path.join(tempdir.name, 'image0.png')
    PIL.Image.fromarray(imagedata).save(imagepath)

    # dont fail
    model.process_image(imagepath, px_per_mm=250)




def test_cells_inference():
    m = ultralytics.YOLO('vendor/ultralytics/ultralytics/cfg/models/26/yolo26.yaml')
    inputsize = m.args['imgsz']
    head   = MaskHead()
    module = CellsYOLO_Module(m, head, px_per_mm=250).eval()
    model  = MaskRCNN_Cells_CARROT(module)


    tempdir = tempfile.TemporaryDirectory()

    # basic, same size as yolo expects
    imagedata = (np.random.random([inputsize, inputsize, 3]) * 255).astype('uint8')
    imagepath = os.path.join(tempdir.name, 'image0.png')
    PIL.Image.fromarray(imagedata).save(imagepath)

    model.process_image(imagepath, px_per_mm=250)


    # bug: odd image size
    imagedata = (np.random.random([489, inputsize, 3]) * 255).astype('uint8')
    imagepath = os.path.join(tempdir.name, 'image0.png')
    PIL.Image.fromarray(imagedata).save(imagepath)

    # dont fail
    model.process_image(imagepath, px_per_mm=250)


def test_stitch_and_relabel_instancemaps_from_grid_bug():
    # 3 x 3 grid
    instancemaps = [
        torch.zeros([100,100], dtype=torch.int32) for i in range(9)
    ]
    # objects in the top-left and top-right maps, but not in top-center
    instancemaps[0][10:20,10:20] = 1
    instancemaps[0][80:90,80:90] = 2
    instancemaps[2][10:20,10:20] = 1
    instancemaps[2][80:90,80:90] = 2
    # no objects in the center row
    # but then again in the bottom row
    instancemaps[6][10:20,10:20] = 1
    instancemaps[6][80:90,80:90] = 2
    instancemaps[8][10:20,10:20] = 1
    instancemaps[8][80:90,80:90] = 2

    grid = torch.as_tensor([
        [0,  0, 100,100],
        [0, 90, 100,190],
        [0,180, 100,280],

        [90,  0, 190,100],
        [90, 90, 190,190],
        [90,180, 190,280],

        [180,  0, 280,100],
        [180, 90, 280,190],
        [180,180, 280,280],
    ]).reshape(3,3,4)

    output0 = stitch_and_relabel_instancemaps_from_grid(instancemaps, grid, slack=10)
    assert len( np.unique(output0.numpy()) ) == len( [0,1,2,3,4,5,6,7,8] )


def test_box_overlap_matrix():
    boxes0 = torch.stack([
        torch.tensor([10,10, 40,40]),
        torch.tensor([10,10, 100,100]),
        torch.tensor([200,200,250,250])
    ])

    output0 = box_overlap_matrix(boxes0, boxes0)
    assert output0.shape == (len(boxes0), len(boxes0))
    # box1 fully overlaps box0
    assert output0[0,1] == 1
    # box0 partially overlaps box1
    assert 0 < output0[1,0] < 1
    assert output0[0,2] == 0



    # empty
    output1 = box_overlap_matrix(boxes0[:0], boxes0)
    assert output1.shape == (0,len(boxes0))
    
    output2 = box_overlap_matrix(boxes0[:0], boxes0[:0])
    assert output2.shape == (0,0)

def test_box_overlap_filter():
    boxes0 = torch.stack([
        torch.tensor([10,10, 40,40]),
        torch.tensor([10,10, 100,100]),
        torch.tensor([200,200,250,250])
    ])

    output0 = box_overlap_filter(boxes0, 0.5)
    assert output0.shape == (len(boxes0),)
    assert output0.dtype == torch.bool

    assert not output0[0]
    assert output0[1]
    assert output0[2]



def test_split_overlaps():
    masks0 = torch.zeros([3, 100,100], dtype=torch.bool)
    masks0[0, 10:30, 10:30] = 1
    masks0[1, 20:50, 20:50] = 1
    masks0[2, 40:90, 20:50] = 1

    output0 = split_overlapping_masks(masks0)
    assert output0.shape == masks0.shape
    assert output0.sum(0).max() == 1
    
    assert (output0[1, 20:45, 30:50] == 1).all()
    assert (output0[2, 45:90, 20:50] == 1).all()

    assert output0[0, 20,20] == 1
    assert output0[0, 24,24] == 1
    assert output0[1, 26,26] == 1
    

    # bug: no overlaps
    masks1 = torch.zeros([3,100,100], dtype=torch.bool)
    masks1[0,10:30, 10:30] = 1
    masks1[1,30:50, 30:50] = 1
    masks1[1,50:90, 50:90] = 1

    output1 = split_overlapping_masks(masks1)
    assert output1.sum(0).max() == 1

    # no masks
    masks2 = torch.zeros([3, 100, 100], dtype=torch.bool)
    output2 = split_overlapping_masks(masks2)
    # just dont fail

    # agent says it fails with one mask:
    masks3 = torch.zeros([1,100,100], dtype=torch.bool)
    masks3[0,10:30, 10:30] = 1

    output3 = split_overlapping_masks(masks3)
    assert output3.sum(0).max() == 1
    # nope no issues


def test_verify_reproducibility(monkeypatch, tmp_path):
    import argparse
    import importlib
    import subprocess
    from src import util

    repo_dir = str(tmp_path / 'repo')
    data_dir = str(tmp_path / 'data')
    edit_dir = str(tmp_path / 'traininglib')
    os.makedirs(repo_dir)
    os.makedirs(data_dir)
    os.makedirs(edit_dir)

    req_file = os.path.join(repo_dir, 'requirements.frozen.txt')
    frozen = (
        'numpy==2.3.3\n'
        '-e git+https://github.com/alexander-g/traininglib'
        '@8921c3f195355c9182069ce82a1febb4e9d7a304#egg=traininglib\n'
    )
    with open(req_file, 'w') as f:
        f.write(frozen)

    versions = {
        os.path.realpath(repo_dir): '2026-11-03:aaaa',
        os.path.realpath(data_dir): '2026-11-03:bbbb',
        os.path.realpath(edit_dir): '2026-11-03:8921c3f195355c9182069ce82a1febb4e9d7a304',
    }
    monkeypatch.setattr(util, 'git_version', lambda p: versions.get(os.path.realpath(p)))
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: frozen)

    class DummyMod:
        __file__ = os.path.join(edit_dir, 'traininglib', '__init__.py')

    monkeypatch.setattr(importlib, 'import_module', lambda name: DummyMod)
    monkeypatch.chdir(repo_dir)

    clean_args = argparse.Namespace(
        trainsplit     = os.path.join(data_dir, 'split.txt'),
        reuse_yolo     = None,
        reuse_maskhead = None,
    )

    # 1. Expected use: clean repo, dataset, freeze, and args
    assert util.verify_reproducibility(clean_args) == []

    # 2. Failure cases: dirty repo, non-repo dataset, dirty dataset, pip mismatch, editable dirty
    versions[os.path.realpath(repo_dir)] = '2026-11-03:aaaa-dirty'
    assert len(util.verify_reproducibility(clean_args)) > 0
    versions[os.path.realpath(repo_dir)] = '2026-11-03:aaaa'

    versions[os.path.realpath(data_dir)] = None
    assert len(util.verify_reproducibility(clean_args)) > 0
    versions[os.path.realpath(data_dir)] = '2026-11-03:bbbb-dirty'
    assert len(util.verify_reproducibility(clean_args)) > 0
    versions[os.path.realpath(data_dir)] = '2026-11-03:bbbb'

    versions[os.path.realpath(edit_dir)] = '2026-11-03:8921c3f-dirty'
    assert len(util.verify_reproducibility(clean_args)) > 0
    versions[os.path.realpath(edit_dir)] = '2026-11-03:8921c3f195355c9182069ce82a1febb4e9d7a304'

    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: 'numpy==1.0.0\n')
    assert len(util.verify_reproducibility(clean_args)) > 0
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: frozen)

    # 3. Edge cases: --reuse-yolo / --reuse-maskhead provided, or missing on treerings parser
    reuse_yolo_args = argparse.Namespace(
        trainsplit     = os.path.join(data_dir, 'split.txt'),
        reuse_yolo     = 'model.pt',
        reuse_maskhead = None,
    )
    assert len(util.verify_reproducibility(reuse_yolo_args)) > 0

    reuse_mh_args = argparse.Namespace(
        trainsplit     = os.path.join(data_dir, 'split.txt'),
        reuse_yolo     = None,
        reuse_maskhead = 'mh.pt.zip',
    )
    assert len(util.verify_reproducibility(reuse_mh_args)) > 0

    treerings_args = argparse.Namespace(
        trainsplit = os.path.join(data_dir, 'split.txt'),
    )
    assert util.verify_reproducibility(treerings_args) == []


def test_per_file_px_per_mm(tmp_path):
    import pytest
    from traininglib import datalib
    from src.maskrcnn_celldetection import InstanceDataset
    from src.treeringmodel import TreeringsDataset
    from src.util import resolve_px_per_mm

    im0 = str(tmp_path / 'im0.png')
    an0 = str(tmp_path / 'an0.png')
    im1 = str(tmp_path / 'im1.png')
    an1 = str(tmp_path / 'an1.png')
    for p in (im0, im1):
        PIL.Image.new('RGB', (100, 100)).save(p)
    for p in (an0, an1):
        PIL.Image.new('L', (100, 100)).save(p)

    splitfile = str(tmp_path / 'split.csv')
    datalib.save_file_tuples(splitfile, [(im0, an0), (im1, an1)])

    pxfile = str(tmp_path / 'px_per_mm.txt')
    with open(pxfile, 'w') as f:
        f.write('im0.png:500\nim1.png:1000\n')

    # 1. Expected use: per-file px_per_mm from file on both datasets
    ds_cells = InstanceDataset.from_splitfile(
        splitfile        = splitfile,
        patchsize        = 50,
        px_per_mm        = pxfile,
        target_px_per_mm = 500,
        cachedir         = str(tmp_path / 'cache_cells'),
    )
    assert len(ds_cells) == 2

    ds_rings = TreeringsDataset.from_splitfile(
        splitfile        = splitfile,
        patchsize        = 50,
        px_per_mm        = pxfile,
        target_px_per_mm = 500,
        cachedir         = str(tmp_path / 'cache_rings'),
    )
    assert len(ds_rings) == 5

    # 2. Failure case: missing file in px_per_mm file
    bad_pxfile = str(tmp_path / 'bad_px.txt')
    with open(bad_pxfile, 'w') as f:
        f.write('im0.png:500\n')
    with pytest.raises(AssertionError):
        resolve_px_per_mm(bad_pxfile, [im0, im1])

    # 3. Edge case: single numeric string or float
    assert resolve_px_per_mm('750', [im0, im1]) == {im0: 750.0, im1: 750.0}
    assert resolve_px_per_mm(750.0, [im0, im1]) == {im0: 750.0, im1: 750.0}


