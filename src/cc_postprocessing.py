import typing as tp

import numpy as np
import PIL.Image
import scipy.ndimage
import skimage.measure as skmeasure
import torch

from .util import load_image, resize_tensor, scale_points_xy


class CellPostprocessingResult(tp.NamedTuple):
    classmap_og:     np.ndarray           # bool, [H,W]
    classmap:        np.ndarray           # bool, [H,W]
    instancemap:     np.ndarray           # int64 [H,W]
    instancemap_rgb: np.ndarray           # uint8 [H,W,3]
    cell_points:     tp.List[np.ndarray]  # list of float64 [N,2], xy

class CombinedPostprocessingResult(tp.NamedTuple):
    cell_info:    tp.List[tp.Dict]
    ringmap_rgb:  np.ndarray         # uint8 [H,W,3]


def postprocess_cellmapfile(path:str, *a, **kw) -> CellPostprocessingResult:
    cellmap = (load_image(path, mode='L') > 127)
    return postprocess_cellmap(cellmap, *a, **kw)


def postprocess_cellmap(
    classmap:  torch.Tensor, 
    workshape: tp.Tuple[int,int],
    og_shape:  tp.Tuple[int,int],
    min_object_size_px: int,
) -> CellPostprocessingResult:
    assert classmap.ndim  == 2, classmap.shape
    assert classmap.dtype == torch.bool

    classmap_og = classmap
    if og_shape != classmap.shape:
        classmap_og = resize_tensor(
            classmap[None].float(), 
            og_shape, 
            mode='nearest'
        )[0].to(classmap.dtype)
    
    if workshape != classmap.shape:
        classmap = resize_tensor(
            classmap[None].float(), 
            workshape, 
            mode='nearest'
        )[0].to(classmap.dtype)
        

    # NOTE: scipy is faster than my own implementation
    instancemap_np, _ = \
        scipy.ndimage.label( classmap.numpy(), structure=np.ones([3,3]) )
    instancemap = torch.as_tensor(instancemap_np)
    instancemap = remove_small_objects(
        instancemap, 
        threshold = min_object_size_px,
    )

    
    instancemap_np  = instancemap.numpy()
    instancemap_rgb = colorize_instancemap(instancemap.numpy())

    cell_points_xy = instancemap_to_cell_points(instancemap.numpy(), og_shape)
    return CellPostprocessingResult(
        classmap_og     = classmap_og.numpy(),
        classmap        = classmap.numpy(),
        instancemap     = instancemap_np,
        instancemap_rgb = instancemap_rgb,
        cell_points     = cell_points_xy,
    )



def postprocess_cells_and_rings_combined(
    cell_points_xy: tp.List[np.ndarray], 
    ring_points_yx: tp.List[tp.Tuple[np.ndarray, np.ndarray]], 
    instancemap:    np.ndarray,
) -> CombinedPostprocessingResult:
    assert instancemap.ndim == 2 and instancemap.dtype in [np.int64, np.int32]

    # NOTE: ring_points in yx format, for legacy reasons
    # converting to xy
    ring_points_xy = [
        (p0[...,::-1], p1[...,::-1]) for p0,p1 in ring_points_yx
    ]

    cell_info = associate_cells(cell_points_xy, ring_points_xy)
    ring_per_cell = np.array([info['year'] for info in cell_info])
    # safeguard if cell_info is empty
    ring_per_cell = ring_per_cell.reshape(-1).astype('int64')

    ringmap_rgb   = colorize_cells_per_ring(instancemap, ring_per_cell)
    return CombinedPostprocessingResult(cell_info, ringmap_rgb)



def instancemap_to_cell_points(
    instancemap: np.ndarray, 
    og_shape:    tp.Optional[tp.Tuple[int,int]] = None,
) -> tp.List[np.ndarray]:
    assert instancemap.ndim == 2
    assert instancemap.dtype in [np.int64, np.int32]

    cell_points_xy = instancemap_to_points(instancemap)
    if og_shape is not None:
        shape:tp.Tuple[int,int] = instancemap.shape # type: ignore
        cell_points_xy = [
            scale_points_xy(p, shape, og_shape) for p in cell_points_xy
        ]
    return cell_points_xy


def instancemap_to_points(
    instancemap: np.ndarray, 
    scale:       float = 1.0
) -> tp.List[np.ndarray]:
    assert instancemap.ndim == 2 and instancemap.dtype in [np.int64, np.int32]

    contours_xy:tp.List[np.ndarray] = []
    for i, prop in enumerate(skmeasure.regionprops(instancemap), start=1):
        propslice:tp.Tuple[slice, slice] = prop.slice
        topleft_xy = propslice[1].start, propslice[0].start
        instancemask = prop.image_convex
        # pad to make sure there are zeros at the borders
        # otherwise will get multiple contours
        instancemask = np.pad(instancemask, pad_width=1)
        contour_yx = skmeasure.find_contours(instancemask, fully_connected='high')
        contour_xy = np.array(contour_yx[0])[...,::-1]
        contour_xy = contour_xy + topleft_xy -1 # -1 because of np.pad
        contour_xy = contour_xy * scale
        contours_xy.append(contour_xy)
    return contours_xy



def associate_cells(
    cell_points: tp.List[np.ndarray], 
    ring_points: tp.List[tp.Tuple[np.ndarray, np.ndarray]], 
) -> tp.List[tp.Dict]:
    '''Assign cells to a ring. All coordinates in xy format.'''
    if len(cell_points) == 0:
        return []
    
    cell_indices = np.cumsum([len(cell) for cell in cell_points])
    all_cell_points = np.concatenate(cell_points)
    # which point is in which ring
    ring_for_points = -np.ones(len(all_cell_points), dtype=np.int64)

    for i,(p0,p1) in enumerate(ring_points):
        polygon = np.concatenate([p0,p1[::-1]], axis=0)
        polygon = skmeasure.approximate_polygon(polygon, tolerance=5)

        in_poly_mask = skmeasure.points_in_poly(all_cell_points, polygon)
        ring_for_points = np.where(in_poly_mask, i, ring_for_points)
    
    cellinfo = []
    for j, ring_ixs in enumerate(np.split(ring_for_points, cell_indices)[:-1]):
        uniques, counts = np.unique(ring_ixs, return_counts=True)
        box_xy = [
            cell_points[j][:,0].min(),
            cell_points[j][:,1].min(),
            cell_points[j][:,0].max(),
            cell_points[j][:,1].max(),
        ]

        cellinfo.append({
            'id':              j,
            'box_xy':          box_xy,
            'year':            int(uniques[counts.argmax()]),
            'area':            polygon_area(cell_points[j]),
            'position_within': 0.0,  # TODO
        })
    return cellinfo

def polygon_area(points: np.ndarray) -> float:
    '''Compute the area of a polygon given its vertices ordered clockwise.
       Shoelace formula.'''
    assert points.ndim == 2 and points.shape[1] == 2
    
    shifted_points = np.roll(points, -1, axis=0)
    # cross product components (determinants)
    cross_products = (points[:, 0] * shifted_points[:, 1] -
                      points[:, 1] * shifted_points[:, 0])
    
    area = 0.5 * np.abs(np.sum(cross_products))
    return area


def colorize_cells_per_ring(instancemap:np.ndarray, ring_per_cell:np.ndarray):
    assert instancemap.ndim == 2 and instancemap.dtype in [np.int64, np.int32]
    assert ring_per_cell.ndim == 1 and ring_per_cell.dtype == np.int64

    COLORS = [
        #(255,255,255),
        ( 23,190,207),
        (255,127, 14),
        ( 44,160, 44),
        (214, 39, 40),
        (148,103,189),
        (140, 86, 75),
        (188,189, 34),
        (227,119,194),
    ]
    GRAY = (224, 224, 224)

    cell_instance_values = np.unique(instancemap)

    rgb = np.zeros(instancemap.shape+(3,), dtype=np.uint8)
    rgb[instancemap > 0] = GRAY
    for i, ring_idx in enumerate(np.unique(ring_per_cell)):
        if ring_idx < 0:
            continue
        cell_ixs  = np.argwhere(ring_per_cell == ring_idx).ravel() + 1
        cell_mask = np.isin(instancemap, cell_instance_values[cell_ixs])
        rgb[cell_mask] = COLORS[i % len(COLORS)]
    return rgb




def remove_small_objects(instancemap:torch.Tensor, threshold:int) -> torch.Tensor:
    assert instancemap.ndim == 2
    assert instancemap.dtype == torch.int64 or instancemap.dtype == torch.int32
    
    labels, counts = torch.unique(instancemap, return_counts=True)
    good_labels = labels[counts > threshold]

    mask = torch.isin(instancemap, good_labels) & (instancemap != 0)
    return mask * instancemap
    


def colorize_instancemap(instancemap:np.ndarray) -> np.ndarray:
    assert instancemap.ndim == 2
    assert instancemap.dtype in [np.int32, np.int64]

    # assuming instance labels are contiguous
    n = instancemap.max()
    hues = np.random.randint(0, 360, size=n)
    sats = np.random.randint(80, 90, size=n)
    vals = np.random.randint(90, 100, size=n)

    colors_hsv = np.concatenate([
        [(0,0,0)],   # black
        np.stack([hues, sats, vals], axis=1)
    ])
    colors_rgb = np.array([ hsv_to_rgb(hsv) for hsv in colors_hsv ])
    instancemap_rgb = colors_rgb[instancemap]
    return instancemap_rgb




def hsv_to_rgb(hsv:tp.Tuple[float,float,float]) -> tp.Tuple[int,int,int]:
    h, s, v = hsv
    # normalize inputs
    h = h % 360
    s = max(0, min(100, s)) / 100.0
    v = max(0, min(100, v)) / 100.0

    if s == 0:
        r = g = b = int(round(v * 255))
        return (r, g, b)

    h_sector = h / 60.0
    i = int(h_sector)  # sector 0..5
    f = h_sector - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    if i == 0:
        r_, g_, b_ = v, t, p
    elif i == 1:
        r_, g_, b_ = q, v, p
    elif i == 2:
        r_, g_, b_ = p, v, t
    elif i == 3:
        r_, g_, b_ = p, q, v
    elif i == 4:
        r_, g_, b_ = t, p, v
    else:  # i == 5
        r_, g_, b_ = v, p, q

    r = int(round(r_ * 255))
    g = int(round(g_ * 255))
    b = int(round(b_ * 255))
    return (r, g, b)


