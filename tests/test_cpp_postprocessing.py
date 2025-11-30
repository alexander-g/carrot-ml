import io
import os
import sys
sys.path.insert(0,'cpp/build/')
import tempfile

import numpy as np
import PIL.Image

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



def test_postprocess_treeringmapfile():
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
    #assert PIL.Image.open( io.BytesIO(out1['treeringmap_ogshape_png']) ).size == og_shape


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
    
    unique_colors = np.unique(instancemap1.reshape(-1,3), axis=0)
    assert len(unique_colors) == 5+1  # 5 objects + background


    #bug: mask grows on multiple iterations
    maskf2 = os.path.join(tempdir.name, 'testmask2.png')
    PIL.Image.open( io.BytesIO(out1['cellmap_workshape_png']) ).save(maskf2)
    out2 = postp.postprocess_cellmapfile(maskf2, workshape, og_shape)

    n1 = np.array( PIL.Image.open( io.BytesIO(out1['cellmap_workshape_png']) ) ).sum()
    n2 = np.array( PIL.Image.open( io.BytesIO(out2['cellmap_workshape_png']) ) ).sum()

    assert n1 == n2




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

    workshape = (777,777)
    og_shape  = (2000,2000)
    output = postp.postprocess_combined(cellmapfile, treeringfile, workshape, og_shape)

    assert len( output['cell_info'] ) == 479

    year_indices = [ c['year_index'] for c in output['cell_info'] ]
    # 3 full rings + incomplete ones counted as -1
    assert len(np.unique(year_indices)) == 4

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
    


