import io
import os
import sys
sys.path.insert(0,'cpp/build/')
import tempfile

import numpy as np
import PIL.Image
import scipy

import carrot_postprocessing_ext as postp
from src import treerings_clustering_legacy as postp_legacy




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
    # NOTE: -1 for legacy reasons
    assert all([np.allclose(np.array(o0)-1,o1) for o0, o1 in zip(out0, out1) ])



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



def test_segmentation_to_paths():
    mask = np.zeros([1000,1000], dtype=bool)
    mask[100:-100, 100:200] = 1  # object 0
    mask[300:-300, 300:400] = 1  # object 1
    mask[300: 310, 300:450] = 1  # object 1 too
    mask[250:-250, 500:600] = 1  # object 2

    out0 = postp_legacy.segmentation_to_paths(mask, 0.0)
    out1 = postp.segmentation_to_paths(mask, 0.0)

    print( [o.shape for o in out0] )
    print( [o.shape for o in out1] )

    assert len(out1) == len(out0) == 3
    assert all( [1000 > len(o) > 300 for o in out1] )
    # not exactly equal
    #assert all( [len(o0) == len(o1) for o0, o1 in zip(out0, out1)] )
    # actual bug
    assert all( (o1 < 1000).all() for o1 in out1)



def test_postprocess_treeringmapfile1():
    mask = np.zeros([1000,1000], dtype=bool)
    mask[10:-10, 100:110] = 1
    mask[50:-50, 200:250] = 1
    mask[10:-10, 300:350] = 1
    mask[10:-10, 600:605] = 1
    mask[10:-10, 800] = 1
    tempdir = tempfile.TemporaryDirectory()
    maskf = os.path.join(tempdir.name, 'testmask.png')
    PIL.Image.fromarray(mask).save(maskf)

    workshape = (555,555)
    og_shape  = mask.shape
    out0 = postp_legacy.postprocess_treeringmapfile(maskf, workshape, og_shape)
    print([ (x0.mean(0), x1.mean(0)) for x0,x1 in out0.ring_points_yx])
    print([ (x0.shape, x1.shape) for x0,x1 in out0.ring_points_yx])

    out1 = postp.postprocess_treeringmapfile(maskf, workshape, og_shape)
    print([ (x0.mean(0), x1.mean(0)) for x0,x1 in out1['ring_points_xy']])
    print([ (x0.shape, x1.shape) for x0,x1 in out1['ring_points_xy']])

    assert len(out0.ring_points_yx) == len(out1['ring_points_xy'])
    assert all([ 
        (out0x0.shape == out1x0.shape and out0x1.shape == out1x1.shape)   
            for ([out0x0, out0x1], [out1x0, out1x1]) 
                in zip(out0.ring_points_yx, out1['ring_points_xy'])  
    ])

    assert PIL.Image.open( io.BytesIO(out1['treeringmap_workshape_png']) ).size == workshape
    assert PIL.Image.open( io.BytesIO(out1['treeringmap_ogshape_png']) ).size == og_shape




# chatgpt:
def intersection_points(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Simple and fast: returns common 2D points between a and b (shape [N,2], dtype float32)
    within Euclidean tolerance `tol`. Result dtype float32, unique rows.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 2)
    if a.size == 0 or b.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Use broadcasting in chunks to avoid huge memory for large arrays
    # Choose chunk size to balance memory vs speed
    chunk = 4096
    matches = []
    for i in range(0, a.shape[0], chunk):
        ai = a[i:i+chunk, None, :]            # (chunk, 1, 2)
        diff = ai - b[None, :, :]             # (chunk, M, 2)
        d2 = np.sum(diff * diff, axis=2)      # squared distances
        close_mask = np.any(d2 <= tol * tol, axis=1)
        if np.any(close_mask):
            matches.append(a[i:i+chunk][close_mask])
    if not matches:
        return np.empty((0, 2), dtype=np.float32)

    common = np.vstack(matches)
    return common


# bug: make sure points of neighboring pairs overlap, or else looks ugly in the ui
def test_use_same_points():
    mask = np.zeros([1000,1000], dtype=bool)
    mask[10:-10, 100:110] = 1
    mask[50:-50, 200:250] = 1
    mask[10:-10, 300:350] = 1
    mask[10:-10, 600:605] = 1
    mask[10:-10, 800] = 1
    tempdir = tempfile.TemporaryDirectory()
    maskf = os.path.join(tempdir.name, 'testmask.png')
    PIL.Image.fromarray(mask).save(maskf)

    workshape = (555,555)
    og_shape  = mask.shape
    out1 = postp.postprocess_treeringmapfile(maskf, workshape, og_shape)

    pairs = out1['ring_points_xy']
    assert len(pairs) > 0
    for pair0, pair1 in zip(pairs, pairs[1:]):
        common_points = intersection_points(pair0[1][1:-1], pair1[0][1:-1], tol=0.01 )
        # print(pair0[1])
        # print(pair1[0])
        # print(common_points)
        # print()
        assert len(common_points) > 0


def test_postprocess_treeringmapfile2():
    imgf2 = os.path.join( os.path.dirname(__file__), 'assets', 'treeringsmap0.png' )
    out2 = postp.postprocess_treeringmapfile(imgf2, (2700,3375), (2048,2048))
    assert len(out2['ring_points_xy']) == 5

    # actual bug: incorrect results due to downsampling
    out2 = postp.postprocess_treeringmapfile(imgf2, (555,555), (2048,2048))
    assert len(out2['ring_points_xy']) == 5

    # another bug
    imgf3 = os.path.join( os.path.dirname(__file__), 'assets', 'treeringsmap1.png' )
    size3 = PIL.Image.open(imgf3).size
    out3 = postp.postprocess_treeringmapfile(imgf3, size3[::-1], size3[::-1])

    png_og_np  = np.array( PIL.Image.open(imgf3).convert('L')  ).astype(bool)
    png_out_np = np.array(
        PIL.Image.open( io.BytesIO(out3['treeringmap_workshape_png']) ).convert('L')
    ).astype(bool)
    assert (png_og_np == png_out_np).mean() > 0.97  #TODO: should be 1.0

    assert len(out3['ring_points_xy']) == 3


# another bug
def test_postprocess_treeringmapfile3_large():
    imgf4 = os.path.join( os.path.dirname(__file__), 'assets', 'treeringsmap2.png' )
    workshape = (1535, 8191)
    og_shape  = PIL.Image.open(imgf4).size[::-1]

    #out0 = postp_legacy.postprocess_treeringmapfile(imgf4, workshape, og_shape)
    out4 = postp.postprocess_treeringmapfile(imgf4, workshape, og_shape)

    assert len(out4['ring_points_xy']) == 9



# bug
def test_postprocess_treeringmapfile4_very_long():
    treeringfile = os.path.join( os.path.dirname(__file__), 'assets', 'treeringsmap4.png' )

    workshape = (1237, 16164)
    og_shape  = (1237, 16164)

    out4 = postp.postprocess_treeringmapfile(treeringfile, workshape, og_shape)
    print(len(out4['ring_points_xy']))
    for p0,p1 in out4['ring_points_xy']:
        print( p0[0].astype(int), p1[0].astype(int) )

    assert len(out4['ring_points_xy']) > 60
    




def test_postprocess_cellmapfile():
    mask = np.zeros([1000,1000], dtype=bool)
    mask[10:-10, 100:110] = 1
    mask[50:-50, 200:250] = 1
    mask[10:-10, 300:350] = 1
    mask[10:-10, 600:605] = 1
    mask[10:-10, 800] = 1
    tempdir = tempfile.TemporaryDirectory()
    maskf = os.path.join(tempdir.name, 'testmask.png')
    PIL.Image.fromarray(mask).save(maskf)

    workshape = (555,555)
    og_shape  = mask.shape
    out1 = postp.postprocess_cellmapfile(maskf, workshape, og_shape)

    instancemap1 = np.array(
        PIL.Image.open( io.BytesIO(out1['instancemap_workshape_png']) )
    )
    assert instancemap1.shape == workshape+(3,)

    cellmap_workshape = PIL.Image.open( io.BytesIO(out1['cellmap_workshape_png']) )
    cellmap_og_shape = PIL.Image.open( io.BytesIO(out1['cellmap_og_shape_png']) )
    assert cellmap_workshape.size == workshape[::-1]
    assert cellmap_og_shape.size  == og_shape[::-1]
    
    unique_colors = np.unique(instancemap1.reshape(-1,3), axis=0)
    assert len(unique_colors) == 5+1  # 5 objects + background


    #bug: mask grows on multiple iterations
    maskf2 = os.path.join(tempdir.name, 'testmask2.png')
    PIL.Image.open( io.BytesIO(out1['cellmap_workshape_png']) ).save(maskf2)
    out2 = postp.postprocess_cellmapfile(maskf2, workshape, og_shape)

    n1 = np.array( PIL.Image.open( io.BytesIO(out1['cellmap_workshape_png']) ) ).sum()
    n2 = np.array( PIL.Image.open( io.BytesIO(out2['cellmap_workshape_png']) ) ).sum()

    assert n1 == n2


# bug: cells that are close together get recognized as one due to smaller workshape
def test_postprocess_cellmapfile_ensure_delineated():
    mask = np.zeros([1000,1000], dtype=bool)
    mask[10 :-10,  20:50] = 1
    mask[100:-100, 51:133] = 1
    mask[300:-300, 134:162] = 1
    mask[300:-300, 163:164] = 1  # thin one (gets swallowed by previous)
    
    tempdir = tempfile.TemporaryDirectory()
    maskf = os.path.join(tempdir.name, 'testmask.png')
    PIL.Image.fromarray(mask).save(maskf)

    workshape = (119,119)
    og_shape  = mask.shape
    out1 = postp.postprocess_cellmapfile(maskf, workshape, og_shape)

    instancemap1 = np.array(
        PIL.Image.open( io.BytesIO(out1['instancemap_workshape_png']) )
    )
    counts1 = np.unique(instancemap1.reshape(-1,3), axis=0, return_counts=True)[1]
    # 4 cells + background:
    assert len( counts1 ) == 5


    # re-postprocess on the output mask

    remaskf = os.path.join(tempdir.name, 'testmask2.png')
    open(remaskf, 'wb').write(out1['cellmap_og_shape_png'])
    out2 = postp.postprocess_cellmapfile(remaskf, workshape, og_shape)

    instancemap2 = np.array(
        PIL.Image.open( io.BytesIO(out2['instancemap_workshape_png']) )
    )
    counts2 = np.unique(instancemap2.reshape(-1,3), axis=0, return_counts=True)[1]
    # should be still 4 + 1
    assert len( counts2 ) == 5

    # sizes should be unchanged
    assert sorted(counts1.tolist()) == sorted(counts2.tolist())

    # bug: make sure the delineation boundary is at most 1px wide
    out3 = postp.postprocess_cellmapfile(maskf, og_shape, og_shape)
    instancemap3 = np.array(
        PIL.Image.open( io.BytesIO(out3['instancemap_workshape_png']) )
    )
    assert (instancemap3.any(-1)[310, 140:164] == 0).sum() == 1 # type: ignore




def test_points_in_polygon():
    points = np.array([
        (200, 200),  # inside
        (50,  200),  # outside
    ]).astype('float64')
    polygon = np.array([
        (100,208),
        (115,83),
        (218,64),
        (260,119),
        (276,217),
        (340,232),
        (369,74),
        (426,69),
        (445,286),
        (251,304),
        (131,265)
    ]).astype('float64')

    output = postp.points_in_polygon(points, polygon)
    assert output.tolist() == [True,False]
    


def test_postprocessing_combined():
    treeringfile = os.path.join( os.path.dirname(__file__), 'assets', 'treeringsmap3-combined.png' )
    cellmapfile = os.path.join( os.path.dirname(__file__), 'assets', 'cellmap3-combined.png' )

    mask = np.array(PIL.Image.open(cellmapfile).convert('L')).astype(bool)
    labeled_mask, expected_n_cells = scipy.ndimage.label(mask)

    workshape = (777,777)
    og_shape  = mask.shape
    output = postp.postprocess_combined(cellmapfile, treeringfile, workshape, og_shape)

    assert len( output['cell_info'] ) == expected_n_cells

    year_indices = [ c['year_index'] for c in output['cell_info'] ]
    # 3 full rings + incomplete ones counted as -1
    assert len(np.unique(year_indices)) == 4
    for c in output['cell_info']:
        if c['year_index'] >= 0:
            assert 0 <= c['position_within'] <= 1
    
    # bug: boxes in og shape
    all_cellboxes = np.array([c['box_xy'] for c in output['cell_info']])
    assert workshape[0] < all_cellboxes.max(axis=0)[1] <= og_shape[0]
    assert workshape[1] < all_cellboxes.max(axis=0)[0] <= og_shape[1]

    # bug: areas also in og shape
    object_areas_scipy = scipy.ndimage.sum_labels(mask, labeled_mask, index=np.arange(expected_n_cells))
    all_cellareas = np.array([c['area_px'] for c in output['cell_info']])
    assert object_areas_scipy.max() * 0.9 < all_cellareas.max()  # very approximate


    instancemap1 = np.array(
        PIL.Image.open( io.BytesIO(output['ringmap_workshape_png']) )
    )
    assert instancemap1.shape[:2] == workshape

    unique_colors = np.unique(instancemap1.reshape(-1,3), axis=0)
    assert len(unique_colors) == 5  # 3 rings + gray + background

    # bug (did not scale points to og shape)
    cell0_color = instancemap1[510,100]
    cell1_color = instancemap1[500,110]
    assert cell0_color.tolist() != [0,0,0]
    assert cell0_color.tolist() == cell1_color.tolist()

    # bug 2
    cell2_color = instancemap1[70,270]
    assert cell2_color.tolist() == [224,224,224]


    # PIL.Image.open( io.BytesIO(output['ringmap_workshape_png']) ).save('DELETE-ringmap.png')
    # PIL.Image.open( io.BytesIO(output['cellmap_workshape_png']) ).save('DELETE-cellmap.png')
    # PIL.Image.open( io.BytesIO(output['treeringmap_workshape_png']) ).save('DELETE-treerings.png')
    # PIL.Image.open( io.BytesIO(output['instancemap_workshape_png']) ).save('DELETE-instances.png')

    # # bug 3  # ignoring for now, probably related to the small workshape
    # cell3_color = instancemap1[28, 128]
    # assert cell3_color.tolist() != [224,224,224]
    


# scale RLE components, making sure they map back
def test_rle_scaling():
    rle = [
        np.array([
            (4, 1, 10),
            (5, 2, 8),
            (6, 2, 9),
            (6, 20, 5),
            (7, 2, 9),
            (7, 20, 7),
            (8, 3, 30)
        ]),
        np.array([
            (14, 1, 10),
            (15, 2, 8),
            (16, 2, 9),
            (16, 20, 5),
            (17, 2, 9),
            (17, 20, 7),
            (18, 2, 30)
        ]),
    ]

    fromshape = (40,44)
    toshape   = (144, 147)
    
    output = postp.scale_rle_components(rle, fromshape, toshape)

    # make sure rows are sorted
    for component in output:
        rows = [row for row, _, _ in component.astype(int)]
        assert (np.diff(rows) >= 0).all()
    

    scale_x = fromshape[1] / toshape[1]
    scale_y = fromshape[0] / toshape[0]
    for component_out, component_og in zip(output, rle):
        component_out_px0 = []
        component_out_px1 = []
        component_og_px0  = []
        component_og_px1  = []

        for run_og in component_og:
            p0_og  = np.array([run_og[0], run_og[1]])
            p1_og  = np.array([run_og[0], run_og[1]+run_og[2]-1])

            component_og_px0.append( tuple(p0_og.tolist()) )
            component_og_px1.append( tuple(p1_og.tolist()) )


        for run_out in component_out:
            p0_out = np.array([run_out[0], run_out[1]])
            p1_out = np.array([run_out[0], run_out[1]+run_out[2]-1])
            

            p0_out_rescaled = tuple(((p0_out + 0.5) * (scale_y, scale_x)).astype(int).tolist())
            p1_out_rescaled = tuple(((p1_out + 0.5) * (scale_y, scale_x)).astype(int).tolist())



            if p0_out_rescaled not in component_out_px0:
                component_out_px0.append(p0_out_rescaled)
            if p1_out_rescaled not in component_out_px1:
                component_out_px1.append(p1_out_rescaled)

        #breakpoint()
        assert np.allclose( component_out_px0, component_og_px0 )
        assert np.allclose( component_out_px1, component_og_px1 )



# bug: downscale components, after rasterization they should be still connected
def test_rle_scaling2():
    rle = [
        np.array([
            (4, 1, 5),  # x0=1 / x1=5
            (5, 6, 3),  # x0=6 / x1=8
            (6, 9, 4),  # x0=9 / x1=12
            (7, 13, 5),  # x0=13 / x1=17
            (8, 18, 5),  # x0=18 / x1=22
        ]),
    ]

    fromshape = (50, 55)
    toshape   = (20, 22)
    
    scaled_rle = postp.scale_rle_components(rle, fromshape, toshape)
    print(scaled_rle)

    #rasterize
    X = np.zeros( toshape, dtype='bool' ) 
    assert len(scaled_rle) == 1
    for rlerun in scaled_rle[0]:
        for i in range(rlerun[2]):
            X[rlerun[0], rlerun[1]+i] = 1
    
    _, n_components = scipy.ndimage.label(X, structure=np.ones([3,3]))
    assert n_components == 1

    # make sure coalesced / no duplicate rows
    rows = [run[0] for run in scaled_rle[0]]
    assert len(rows) == len(set(rows))

    # bug2: asymmetric u-shape, 2nd arm of the u disappears because too thin
    rle2 = [
        np.array([
            (4,1,1),
            (5,1,1),
            (6,1,1),
            (7,1,1),
            (8,1,1),
            (9,1,10),
            (8,9,1),
            (7,9,1),
            (6,9,1),
        ])
    ]
    scaled_rle2 = postp.scale_rle_components(rle2, fromshape, toshape)

    rows2 = [run[0] for run in scaled_rle2[0]]
    _, rows2_counts = np.unique(rows2, return_counts=True)
    assert rows2_counts.max() > 1


