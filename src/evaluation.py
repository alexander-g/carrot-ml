import typing as tp

import torch


def IoU_matrix(a:torch.Tensor, b:torch.Tensor, zero_out_zero:bool) -> torch.Tensor:
    # assuming values are sequential

    assert a.ndim == b.ndim == 2
    assert a.shape == b.shape
    assert a.dtype == b.dtype == torch.int32

    uniques_a, counts_a = torch.unique(a, return_counts=True)
    uniques_b, counts_b = torch.unique(b, return_counts=True)
    matrix = torch.empty([len(uniques_a), len(uniques_b)], dtype=torch.int64)

    # stack and view as int64 for faster processing
    stacked = torch.stack([a,b], dim=-1).view(torch.int64)
    uniques_ab, counts_ab = torch.unique(stacked, return_counts=True)

    # back to int32 and labels that overlap are paired [N,2]
    unstacked = uniques_ab.view(torch.int32).reshape(-1,2)

    intersections_mat = torch.zeros_like(matrix)
    intersections_mat[unstacked[:,0], unstacked[:,1]] = counts_ab

    unions_mat =  torch.zeros_like(matrix)
    unions_mat += counts_a[:,None]
    unions_mat += counts_b[None,:]
    unions_mat = unions_mat - intersections_mat

    iou_mat = intersections_mat / unions_mat
    iou_mat[unions_mat == 0] = 0

    if zero_out_zero:
        iou_mat[uniques_a == 0, :] = 0
        iou_mat[:, uniques_b == 0] = 0

    return iou_mat



def best_iou_matches_greedy(
    iou_matrix: torch.Tensor,
    threshold:  float,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    ''' Given an IoU matrix of shape [N, M], get label index pairs with the
        highest overlap using a greedy algorithm (one-to-one matching) '''
    assert iou_matrix.ndim == 2

    iou_matrix = iou_matrix.clone()
    matches    = []
    iou_values = []
    while iou_matrix.numel() > 0:
        max_index = iou_matrix.argmax()

        if iou_matrix.ravel()[max_index] < threshold:
            break

        max_i = int( max_index // iou_matrix.shape[1] )
        max_j = int( max_index % iou_matrix.shape[1] )
        matches.append( (max_i, max_j) )
        iou_values.append(float(iou_matrix[max_i, max_j]))

        # entire row and column corresponding to the matched pair to zero
        iou_matrix[max_i, :] = 0
        iou_matrix[:, max_j] = 0
    
    return torch.tensor(matches), torch.tensor(iou_values)

