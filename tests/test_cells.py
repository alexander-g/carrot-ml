#from src.cellsmodel import relabel_instancemaps
from src.maskrcnn_celldetection import relabel_instancemaps

import torch



def test_relabel_instancemaps():
    x0 = torch.zeros([100,100], dtype=torch.int64)
    x1 = torch.zeros([100,100], dtype=torch.int64)

    x0[20:30,20:30] = 1   # no overlaps
    x0[20:30,80:  ] = 2   # overlapping with x1:3

    x1[50:60,50:60] = 1   # no overlaps
    x1[20:30, 0:30] = 3   # overlapping with x0:2

    new_x1 = relabel_instancemaps(x0,x1, (90,0,10,100), (0,0,10,100) )

    assert (new_x1[x1==3] == 2).all()
    assert (new_x1[x1==1] != 1).all()


def test_relabel_instancemaps_filter_out_small_overlaps():
    x0 = torch.zeros([100,100], dtype=torch.int64)
    x1 = torch.zeros([100,100], dtype=torch.int64)

    x0[20:30,20:30] = 1   # no overlaps
    x0[20:30,80:95] = 2   # single-pixel overlap with x1:3, but later

    x1[50:60,50:60] = 1   # no overlaps
    x1[20:30, 5:30] = 3   # no overlaps yet with x0:2

    new_x1 = relabel_instancemaps(x0,x1, (90,0,10,100), (0,0,10,100), minimum_overlap_pixels=0 )
    assert not set(torch.unique(new_x1)[1:].numpy()).intersection(torch.unique(x0).numpy()), 'no overlaps here yet'
    print(torch.unique(new_x1, return_counts=True))
    
    x1[25, 4:30] = 3   # single pixel overlap with x0:2
    new_x1 = relabel_instancemaps(x0,x1, (90,0,10,100), (0,0,10,100), minimum_overlap_pixels=2 )
    assert not set(torch.unique(new_x1)[1:].numpy()).intersection(torch.unique(x0).numpy()), 'single pixel overlap should have been ignored'
    assert len(torch.unique(new_x1)) == 3   # 2+background
    print(torch.unique(new_x1, return_counts=True))

