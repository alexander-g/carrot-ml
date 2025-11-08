import sys
sys.path.insert(0,'cpp/build/')

import numpy as np

import carrot_postprocessing_ext as postp
from src import treerings_clustering_legacy as postp_legacy



def test_merge_paths():
    paths0 = [
        np.linspace([10,10], [30,30], 20),
        np.linspace([110,10], [120,20], 10),
        # should be merged with #0
        np.linspace([50,50], [60,60], 10),
    ]
    imageshape = (200,200)


    out0 = postp_legacy.merge_paths(paths0, imageshape)
    out1 = postp.merge_paths(paths0, imageshape)

    assert len(out0) == 2
    assert all([np.allclose(o0,o1) for o0, o1 in zip(out0, out1) ])



def test_associate_boundaries():
    paths0 = [
        np.linspace([10,10],  [90,90], 20),
        np.linspace([210,10], [290,90], 20),
        np.linspace([110,10], [190,90], 20),
    ]

    out0 = postp_legacy.associate_boundaries(paths0)
    out1 = postp.associate_boundaries(paths0)

    print(out0)
    print(out1)
    assert len(out0) == len(out1)
    assert all([np.allclose(o0,o1) for o0, o1 in zip(out0, out1) ])



def test_associate_pathpoints():
    path0 = np.linspace([ 10,10], [150,150], 30)
    path1 = np.linspace([210,10], [300,100], 30)

    out0p0, out0p1 = postp_legacy.associate_pathpoints(path0, path1)
    out1p0, out1p1 = postp.associate_pathpoints(path0, path1)
    
    print(out0p0.shape, out0p1.shape)
    print(out1p0.shape, out1p1.shape)
    assert out1p0.shape == out1p1.shape
    assert out1p0.shape == out0p0.shape

    assert all([np.allclose(o0,o1) for o0, o1 in zip(out1p0, out0p0) ])
    assert all([np.allclose(o0,o1) for o0, o1 in zip(out1p1, out0p1) ])


