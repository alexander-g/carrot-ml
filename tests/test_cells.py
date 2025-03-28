from src.cellsmodel import relabel_instancemaps

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



