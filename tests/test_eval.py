from src.evaluation import IoU_matrix

import torch


def test_iou_matrix_basic():
    a = torch.tensor([
        [1, 1, 2],
        [1, 2, 2],
        [3, 3, 0]
    ], dtype=torch.int32)

    b = torch.tensor([
        [1, 2, 2],
        [1, 2, 0],
        [3, 3, 0]
    ], dtype=torch.int32)

    iou = IoU_matrix(a, b)

    expected = torch.tensor([
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 2/3, 1/5, 0.0],
        [1/4, 0.0, 2/4, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    print(iou)
    assert torch.allclose(iou, expected)



