import typing as tp

import numpy as np
import scipy
import skimage
import PIL.Image

import skimage.morphology as skmorph
import skimage.measure
import skimage.draw
import skimage.graph        as skgraph
import scipy.sparse.csgraph as csgraph


#Helper functions for tree ring clustering and measurement


def tree_ring_clustering(x:np.ndarray, **kw):
    '''Find tree ring boundaries in a segmentation output x'''
    assert x.ndim == 2
    paths        = segmentation_to_paths( x > 0 )
    merged_paths = merge_paths(paths, x.shape, **kw)  # type: ignore
    return merged_paths



def furthest_point(points:np.ndarray, p:np.ndarray) -> tp.Optional[np.ndarray]:
    '''Get the points which has the maximum distance from p'''
    if len(points) == 0:
        return None
    
    i = scipy.spatial.distance_matrix(points, p[np.newaxis]).argmax()
    return points[i]


def get_endpoints_of_cluster(
    cluster_points:np.ndarray
) -> tp.Optional[ tp.Tuple[np.ndarray, np.ndarray] ]:
    '''Get two points in a cluster that are the furthest away from each other (approximate)'''
    if len(cluster_points) == 0:
        return None
    
    a0 = np.array([0,0])   #zero to semi-ensure points being ordered
    a1 = furthest_point(cluster_points, a0)
    a2 = furthest_point(cluster_points, a1) # type: ignore
    return a2,a1 # type: ignore


def closest_point(
    points: np.ndarray, 
    p:      np.ndarray, 
    d:      float = 0.0
) -> tp.Optional[np.ndarray]:
    '''Get the closest point to p at distance d'''
    if len(points) == 0:
        return None
    
    dmat = scipy.spatial.distance_matrix(points, p[np.newaxis])
    dmat = np.abs(dmat - d)
    i    = dmat.argmin()
    return points[i]


def get_neighborhood(
    points:    np.ndarray, 
    reference: np.ndarray,  
    distance:  float
) -> np.ndarray:
    '''Get points that are at most `distance` away from the reference point'''
    #distances = np.sqrt( np.sum((points - reference)**2, axis=-1) )
    dmat = scipy.spatial.distance_matrix(points, reference[np.newaxis])[:,0]
    return points[ dmat < distance ]


def normalize(x:np.ndarray) -> np.ndarray:
    '''Normalize x to unit length'''
    xlen = np.sum(x**2)**0.5
    return x/np.maximum(xlen, 1e-6)


def eval_implicit_equation(y, x, coef):
    '''Evaluate the equation ax + by = c
       i.e. the distance of `(x,y)` to the line represented as a implicit equation `coef`'''
    (a,b),c = normalize(np.array(coef[1:3])), coef[0]
    return c + y*a + x*b


def line_from_two_points(a0:np.ndarray, a1:np.ndarray) -> tp.List:
    '''Compute the coefficients of a line going through the points `a0` and `a1`'''
    direction = normalize( a0-a1 )
    ortho     = [ direction[1], -direction[0] ]
    offset    = -np.sum(a0 * ortho)
    coef      = [ offset ] + ortho
    return coef


def line_from_endpoint(
    p:         np.ndarray, 
    cluster:   np.ndarray, 
    threshold: float = 200,
) -> tp.Optional[tp.List]:
    '''Compute the coefficients of a line going through an endpoint `p` 
       of a cluster of points'''
    nhood = get_neighborhood(cluster, p, threshold)
    a0    = np.mean(nhood,0)
    a1    = furthest_point(nhood, p)
    if a1 is None:
        return None
    return line_from_two_points(a0, a1)


def rotate_ccw(coef:tp.List, p:np.ndarray) -> tp.List:
    '''Rotate a line by 90° counter-clockwise so that it goes through point `p`'''
    ortho_inv = [ -coef[2], coef[1] ]
    offset    = -np.sum(p * ortho_inv)
    return [offset] + ortho_inv


def get_points_in_ray(coef, p, points, threshold=50, distance=np.inf):
    '''Compute which points are within `threshold` from line 
       and `distance` away in front of `p`.'''
    y,x = points[:,0], points[:,1]
    coef_inv          = rotate_ccw(coef, p)
    point_distances   = eval_implicit_equation(y, x, coef_inv)
    # points close to line
    in_line_mask      = np.abs(eval_implicit_equation(y, x, coef)) < threshold 
    # points in front of p
    in_direction_mask = point_distances > 0
    # points within distance of p
    in_distance_mask = point_distances < distance
    in_ray_mask      = in_line_mask & in_direction_mask & in_distance_mask
    return in_ray_mask


def sort_clusters(clusters:tp.List[np.ndarray]) -> tp.List[np.ndarray]:
    '''Sort clusters along x or y axis (whichever is the longest in the majority)'''
    clusters = [c for c in clusters if len(c)]

    a0 = np.array([0,0])
    A1 = [furthest_point(c, a0) for c    in clusters]
    A2 = [furthest_point(c, a1) for c,a1 in zip(clusters, A1)] # type: ignore
    
    longest_axes = np.array([ 
        np.argmax( np.abs(a2-a1) )for a1,a2 in zip(A1,A2)] # type: ignore
    )
    common_axis  = int(np.median(longest_axes))
    clusters     = [ c[np.argsort(c[:,common_axis])] for c in clusters ]
    return clusters



def sample_points(points:np.ndarray, n:int) -> np.ndarray:
    '''Select n roughly equidistant points'''
    a0,a1 = points[0], points[-1]  #assuming sorted
    d = scipy.spatial.distance.euclidean(a0,a1)
    D = np.linspace(0, d, n)
    return np.array( [closest_point(points, a0, d) for d in D] )

def find_next_boundary(
    points: np.ndarray, 
    labels: np.ndarray, 
    l: int, 
    reverse: bool=False,
) -> tp.Optional[int]:
    '''Robustly determine the next tree ring boundary relative to the boundary `l`'''
    #points belonging to specified ring
    cluster      = points[labels==l]
    #points belonging to all other rings (also no outliers)
    other_points = points[(labels!=l) * (labels!=0)]
    other_labels = labels[(labels!=l) * (labels!=0)]
    
    sampled_points = sample_points(cluster, n=25+1)
    sampled_labels = []
    sampled_pairs  = []
    for i in range(len(sampled_points)-1):
        p0 = sampled_points[i]
        p1 = sampled_points[i+1]
        if reverse:
            p0,p1 = p1,p0
        #fit a line
        coef = line_from_two_points(p0, p1)
        #rotate it by 90 deg
        coef_orto  = rotate_ccw(coef, p0)
        #get points from other lines that intersect the 90deg line
        intersection = get_points_in_ray(coef_orto, p0, other_points, threshold=25)
        intersected  = other_points[intersection]
        intersection_labels = other_labels[intersection]
        if len(intersected) > 0:
            #get the index of the closest point in intersection
            distances = scipy.spatial.distance_matrix(intersected, p0[np.newaxis])
            closest_i = distances.argmin()
            #get the closest point and its label
            sampled_pairs  += [(p0, intersected[closest_i])]
            sampled_labels += [intersection_labels[closest_i]]
    
    if len(sampled_labels)==0:
        return None
    #take the most common label
    unique_l, unique_counts = np.unique(sampled_labels, return_counts=True)
    next_l = unique_l[unique_counts.argmax()]
    return next_l

def associate_boundaries(paths:tp.List[np.ndarray]) -> tp.List:
    '''Group tree ring boundaries into tuples'''
    if len(paths) == 0:
        return []
    points = np.concatenate(paths)
    labels = np.concatenate([np.ones(len(p), int)*i+1 for i,p in enumerate(paths)])
    pairs = []
    for i,l in enumerate(np.unique(labels)):
        if l==0:
            continue
        next_l = find_next_boundary(points, labels, l)
        if next_l is None:
            continue
        #cycle consistency
        prev_l = find_next_boundary(points, labels, next_l, reverse=True)
        if l == prev_l:
            pairs.append((l,next_l))
    labels_in_pairs, label_counts = np.unique(pairs, return_counts=True)
    endpoints = labels_in_pairs[label_counts==1]
    #sort the boundaries, closest to the topleft corner first
    endpoints = sorted(
        endpoints, 
        key=lambda l: (np.sum(points[labels==l]**2,-1)**0.5).mean() if l!=0 else np.inf
    )
    
    chains = []
    for e in endpoints:
        next_l  = e
        chain   = []
        _pairs  = list(pairs)
        while 1:
            for i, pair in enumerate(_pairs):
                if next_l == pair[1]:
                    pair = pair[::-1]
                if next_l == pair[0]:
                    chain.append(pair)
                    next_l = pair[1]
                    _pairs.pop(i)
                    break
            else:
                break
        chains.append(chain)
    #take the longest chain
    if len(chains)==0:
        return []
    longest_chain = chains[np.argmax([len(c) for c in chains])]
    return longest_chain

# def associate_points(points0, points1, sort=True):
#     '''Group points from boundary 0 to corresponding points from boundary 1'''
#     #reduce the number of points, for speed reasons
#     step = int(np.ceil(max(len(points0), len(points1))/1500))
#     points0, points1 = points0[::step], points1[::step]
#     if sort:
#         p0      = get_endpoints_of_cluster(points0)[0]
#         points0 = points0[ np.argsort(((points0 - p0)**2).sum(-1)) ]
#     dmat    = scipy.spatial.distance_matrix(points0, points1)
#     matches = scipy.optimize.linear_sum_assignment(dmat)
#     points0 = points0[matches[0]]
#     points1 = points1[matches[1]]
#     return points0, points1


# TODO: refactor
def associate_cells_from_segmentation(
    cell_map:    np.ndarray, 
    ring_points: tp.List[tp.Tuple[np.ndarray, np.ndarray]], 
    og_size:     tp.Optional[tp.Tuple[int,int]] = None,
    min_area:    float=32,
):
    '''Assign a tree ring label to each cell'''
    assert cell_map.ndim == 2


    # draw tree rings polygons to create a ring_map
    # then crop the ring_map at the cell locations to get the cell's label
    # intermediate downscaling for faster processsing
    _scale = 4
    H,W    = cell_map.shape

    # if the cell_map is scaled down
    og_scale = (1.0, 1.0)
    if og_size is not None:
        og_scale = (W/og_size[0], H/og_size[1])

    ring_map = np.zeros(np.array([H,W])//_scale, 'int16')
    for i,(p0,p1) in enumerate(ring_points):
        polygon = np.concatenate([p0,p1[::-1]], axis=0) / _scale
        polygon = skimage.measure.approximate_polygon(polygon, tolerance=5)
        polygon_pixel_coords = \
            skimage.draw.polygon( polygon[:,0], polygon[:,1], ring_map.shape )
        ring_map[polygon_pixel_coords] = (i+1)
    #upscale to the original size
    ring_map = PIL.Image.fromarray(ring_map).resize([W,H], PIL.Image.NEAREST)
    ring_map = (ring_map * cell_map).astype(np.int16)
    ring_map_rgb = np.zeros(ring_map.shape+(3,), 'uint8')
    within_map   = create_within_map(ring_points, (H,W), scale=_scale)
    
    COLORS = [
        (255,255,255),
        ( 23,190,207),
        (255,127, 14),
        ( 44,160, 44),
        (214, 39, 40),
        (148,103,189),
        (140, 86, 75),
        (188,189, 34),
        (227,119,194),
    ]
    
    labeled_cells = scipy.ndimage.label(cell_map)[0]
    cells         = []
    for i,slices in enumerate(scipy.ndimage.find_objects(labeled_cells)):
        cell_mask = (labeled_cells[slices] == (i+1))
        area      = int(cell_mask.sum())
        if area < min_area:
            continue
        
        cell_labels, counts = \
            np.unique(ring_map[slices][cell_mask], return_counts=True)
        if len(counts) == 0:
            continue

        max_label = cell_labels[counts.argmax()]
        ring_map_rgb[slices][cell_mask] = COLORS[max_label%len(COLORS)]
        
        box_xy = [
            slices[1].start / og_scale[0], 
            slices[0].start / og_scale[1], 
            slices[1].stop  / og_scale[0], 
            slices[0].stop  / og_scale[1],
        ]
        position_within: tp.Optional[float] = None
        if within_map is not None:
            position_within = \
                cell_position_within(box_xy, within_map, max_label, _scale)
        cells.append({
            'id':              i,
            'box_xy':          box_xy,
            'year':            int(max_label),
            'area':            area,
            'position_within': position_within,
        })
    return cells, ring_map_rgb

def create_within_map(
    ring_points: tp.List[tp.Tuple[np.ndarray, np.ndarray]], 
    imgshape:    tp.Tuple[int,int], 
    scale:       float = 1.0
) -> tp.Optional[np.ndarray]:
    '''Computes a map which encodes the position of pixels within a tree ring.
       - shape: [n_rings, H, W]
       - range: 0-100, invalid where negative
       - `scale` for speed
    '''
    yx = np.stack(np.meshgrid( 
        np.arange(imgshape[0]/scale, dtype='int32'), 
        np.arange(imgshape[1]/scale, dtype='int32'), 
        indexing='ij' 
    ), axis=-1)
    
    withins = [
        scipy.interpolate.griddata(
            np.concatenate([p0,p1])/scale,
            np.concatenate([np.zeros(len(p0)), 100*np.ones(len(p1))]), 
            yx,
            fill_value=-99
        ).astype('int8') for p0,p1 in ring_points  #int8 to save memory
    ]
    return np.stack(withins) if len(withins) else None

def cell_position_within(
    cellbox:    tp.List, 
    within_map: np.ndarray, 
    ringlabel:  int, 
    scale:      float = 1.0,
) -> tp.Optional[float]:
    '''Returns the approximate position of a cell within a tree ring.
       - `scale` should be the same as used in create_within_map()
    '''
    if ringlabel==0:
        return None
    cellbox = (np.array(cellbox)/scale).astype(int) # type: ignore
    crop    = within_map[ringlabel-1, cellbox[1]:cellbox[3], cellbox[0]:cellbox[2]]
    return crop[crop>=0].mean() if np.any(crop>=0) else None




def reorient_paths(paths:tp.List[np.ndarray]) -> tp.List[np.ndarray]:
    if len(paths) == 0:
        return []
    
    directions = np.array([ (p[-1] - p[0]) for p in paths])
    max_axes   = np.argmax(np.abs(directions), -1)
    ax_unq, ax_cnts = np.unique(max_axes, return_counts=True)
    common_axis     = ax_unq[ax_cnts.argmax()]

    orientations  = np.sign( directions[:, common_axis] )
    o_unq, o_cnts = np.unique(orientations, return_counts=True)
    common_o      = o_unq[o_cnts.argmax()]
    new_paths     = [ 
        p if o == common_o else p[::-1] for p,o in zip(paths, orientations) 
    ]
    return new_paths

def single_skeleton_to_path(skeleton_image:np.ndarray) -> np.ndarray:
    graph, nodes = skgraph.pixel_graph(skeleton_image, connectivity=2)
    #node mask that indicates which nodes are endpoints
    endpoints = np.array(((graph>0).sum(0)==1).tolist()[0])
    shortests, preds = csgraph.shortest_path(
        graph, 
        indices  = np.argwhere(endpoints).ravel(), 
        directed = False, 
        return_predecessors = True,
    )
    if len(shortests) < 1:
        return np.empty([0,2])
    _i = shortests.max(1).argmax()
    i  = endpoints[_i]
    j  = shortests[_i].argmax()
    
    path_ixs  = [preds[_i][j]]
    while path_ixs[-1] >= 0:
        path_ixs += [ preds[_i][path_ixs[-1]] ]
    path_ixs = np.array(path_ixs[::-1][1:], 'int')  # type: ignore
    path    =  np.unravel_index( nodes[path_ixs], skeleton_image.shape )
    return np.stack(path, -1)

def segmentation_to_paths(x:np.ndarray, min_length:float=0.0) -> tp.List[np.ndarray]:
    '''Find connected components in a segmentation map and convert them to paths'''
    assert x.dtype == np.bool_
    
    skeleton = skmorph.skeletonize(x)
    labeled  = skmorph.label(skeleton)
    paths    = []
    for l,slices in enumerate(scipy.ndimage.find_objects(labeled), 1):
        path    = single_skeleton_to_path(labeled[slices]==l)
        if len(path) < 2:
            continue
        path   += [slices[0].start, slices[1].start]
        paths  += [path]
    
    if min_length < 1.0:
        #relative to image width (normally the smaller side of an image)
        min_length = min(x.shape)*min_length
    paths = [p for p in paths if len(p)>=min_length and len(p)>1]
    paths = reorient_paths(paths)
    return paths


def project_points_onto_line(coef:tp.List, points:np.ndarray) -> np.ndarray:
    y,x = points[:,1], points[:,0]
    signed_distances = eval_implicit_equation(y, x, coef)
    direction        = normalize(np.array(coef[1:3]))
    return points + direction * -signed_distances[:,None]

def projected_points_to_1d(proj_points:tp.List[np.ndarray]) -> tp.Optional[tp.List]:
    '''Convert sets of 2d points that lie on the same line to 1d points'''
    if len(proj_points) == 0:
        return None
    list_of_endpoints = list(map(get_endpoints_of_cluster, proj_points))
    anchor = get_endpoints_of_cluster(np.concatenate(list_of_endpoints)) # type: ignore
    if anchor is None:
        return None
    distances = [
        scipy.spatial.distance.cdist(anchor[:1], ep)[0] for ep in list_of_endpoints
    ]
    return distances
    
def overlap_1d(points0:np.ndarray, points1:np.ndarray) -> np.ndarray:
    '''Compute how much points0 is overlapped by points1 (1D points)'''
    assert len(np.shape(points0)) == len(np.shape(points1)) == 1
    isec_start = np.max([np.min(points0), np.min(points1)])
    isec_end   = np.min([np.max(points0), np.max(points1)])
    return np.maximum(isec_end - isec_start, 0) / (np.max(points0) - np.min(points0))
    
def max_overlap_on_line(
    coef:    tp.List, 
    points0: np.ndarray, 
    points1: np.ndarray,
) -> tp.Optional[float]:
    '''Compute how much two sets of points overlap each other if projected onto a line'''
    points_on_line_0 = project_points_onto_line(coef, points0)
    points_on_line_1 = project_points_onto_line(coef, points1)
    points_1d = projected_points_to_1d([points_on_line_0, points_on_line_1])
    if points_1d is None:
        return None
    p0, p1 = points_1d
    return max(overlap_1d(p0, p1), overlap_1d(p1, p0)) # type: ignore

def max_mutual_overlap(path0:np.ndarray, path1:np.ndarray) -> float:
    '''Approximate how much two paths overlap if projected onto each other'''
    coef0 = line_from_two_points(path0[0], path0[-1])
    coef1 = line_from_two_points(path1[0], path1[-1])
    path0_on_line_0 = project_points_onto_line(coef0, path0)
    path1_on_line_0 = project_points_onto_line(coef0, path1)
    path0_on_line_1 = project_points_onto_line(coef1, path0)
    path1_on_line_1 = project_points_onto_line(coef1, path1)

    p0_1d = projected_points_to_1d([path0_on_line_0, path1_on_line_0])
    p1_1d = projected_points_to_1d([path1_on_line_1, path0_on_line_1])
    return max( # type: ignore
        overlap_1d(*p0_1d), # type: ignore
        overlap_1d(*p1_1d), # type: ignore
    )

def get_paths_in_ray(coef, e, paths, *args, **kw) -> tp.List:
    if len(paths) == 0:
        return []
    points    = np.concatenate(paths)
    labels    = np.concatenate([np.ones(len(p), int)*i for i,p in enumerate(paths)])
    inraymask = get_points_in_ray(coef, e, points, *args, **kw)
    return [paths[l] for l in np.unique(labels[inraymask])]

def merge_and_reorder(path0, path1):
    D   = scipy.spatial.distance.cdist([path0[0], path0[-1]], [path1[0], path1[-1]])
    i   = D.argmin()
    return ( np.concatenate([ path1[::-1], path0 ])       if i==0 else
             np.concatenate([ path1      , path0 ])       if i==1 else 
             np.concatenate([ path0,       path1 ])       if i==2 else
             np.concatenate([ path0,       path1[::-1] ]) )

def select_path_closest_to_point(p, paths):
    min_dists = [scipy.spatial.distance.cdist([p], path).min() for path in paths]
    return paths[np.argmin(min_dists)]


def merge_paths(
    paths:      tp.List[np.ndarray], 
    imageshape: tp.Tuple[int,int], 
    max_distance = 0.30, 
    ray_width    = 50, 
    max_overlap  = 0.3, 
    min_length   = 0.05
) -> tp.List[np.ndarray]:
    '''Merge a set of paths into tree ring boundaries
       1. **Iterate over spaths in descending order**
       2. **Fit a ray on each of the endpoints**
       3. **Merge other components that intersect with the ray within a threshold
            if they dont overlap too much onto each other.**
    '''
    if max_distance < 1:
        #relative to image width (normally the smaller side of an image)
        max_distance *= min(imageshape)
    if min_length < 1:
        #relative to image width (normally the smaller side of an image)
        min_length *= min(imageshape)
    
    paths = [np.copy(p) for p in paths]
    paths = sorted(paths, key=len, reverse=True)
    for i,path in enumerate(paths):
        if len(path) < min_length:
            #discard small paths
            path.resize([0,2], refcheck=False)
            continue
        
        endpoints = [path[0], path[-1]]
        while len(endpoints):
            e = endpoints.pop(0)
            isec_paths = []
            for d in [len(path)*0.1, len(path)*0.2, len(path)*1.0]:
                coef       = line_from_endpoint(e, path, d)
                #get paths that intersect the ray within a threshold
                isec_paths += get_paths_in_ray(coef, e, paths, distance=max_distance)
            #remove candidates that have a large overlap
            isec_paths = [p for p in isec_paths if max_mutual_overlap(path, p) < max_overlap]
            if len(isec_paths) == 0:
                continue
            merge_path = select_path_closest_to_point(e, isec_paths)
            #merge it to the previous one
            path       = merge_and_reorder(path, merge_path)
            paths[i]   = path
            #current path has changed: repeat iteration
            endpoints  = [path[0], path[-1]]
            
            #set size to zero to indicate that this path has been processed
            merge_path.resize([0,2], refcheck=False)
    paths = [p for p in paths if len(p)]
    return paths

def distances(x,y):
    return ((x - y)**2).sum(-1)**0.5

def path_distances(path):
    return distances(path[1:], path[:-1])

def path_length(path):
    return path_distances(path).sum()

def associate_pathpoints(path0, path1) -> tp.Tuple[np.ndarray, np.ndarray]:
    '''Group points from path 0 to corresponding points from path 1'''
    u0       = np.cumsum(path_distances(path0))
    u0       = np.concatenate([[0],u0])
    tck0, u0 = scipy.interpolate.splprep(path0.T, k=1, u=u0)
    
    u1       = np.cumsum(path_distances(path1))
    u1       = np.concatenate([[0],u1])
    tck1, u1 = scipy.interpolate.splprep(path1.T, k=1, u=u1)
    
    #resample the path equidistantly
    step     = 64
    t0       = np.concatenate( [np.arange(0, u0[-1], step), u0[-1:]] )
    path0    = np.stack(scipy.interpolate.splev(t0, tck0),-1)
    t1       = np.concatenate( [np.arange(0, u1[-1], step), u1[-1:]] )
    path1    = np.stack(scipy.interpolate.splev(t1, tck1),-1)
    
    flipped  = len(path0) < len(path1)
    if flipped:
        path0, path1 = path1, path0
    
    #find the best offset
    meandists = [
        distances(path0[i:][:len(path1)], path1).mean() 
            for i in range(len(path0) - len(path1) +1)
    ]
    offset    = np.argmin(meandists)
    path0     = path0[offset:][:len(path1)]
    
    if flipped:
        path0, path1 = path1, path0
        
    return path0, path1


def treering_area(borderpoints0, borderpoints1) -> float:
    '''Compute the area of the polygon defined by treering border points'''
    triangles0 = np.stack([
        borderpoints0[:-1], borderpoints0[1:], borderpoints1[:-1]
    ], axis=1)
    triangles1 = np.stack([
        borderpoints1[:-1], borderpoints1[1:], borderpoints0[:-1]
    ], axis=1)
    triangles = np.concatenate([triangles0, triangles1])
    
    area = 0.0
    for tri in triangles:
        (p0,p1,p2) = tri[0], tri[1], tri[2]
        coef = line_from_two_points(p0,p1)
        p2y, p2x = p2[1], p2[0]
        h    = np.abs(  eval_implicit_equation(p2y, p2x, coef)  )
        b    = scipy.spatial.distance.euclidean( p0,p1 )
        area += b*h/2
    return area
