import os
import tempfile

from src.cells_yolo import CellsYOLO_Module, CellsYOLO_Module, MaskRCNN_Cells_CARROT
from src.cells_yolo_maskhead import MaskHead
from src.treerings_yolo import TreeringsYOLO_Module, Treerings_CARROT, TreeringsInference
from src.maskrcnn_celldetection import stitch_and_relabel_instancemaps_from_grid

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


