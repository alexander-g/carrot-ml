import os
import tempfile

from src.cells_yolo import CellsYOLO_Module, CellsYOLO_Module, MaskRCNN_Cells_CARROT
from src.treerings_yolo import TreeringsYOLO_Module, Treerings_CARROT, TreeringsInference

import numpy as np
import ultralytics
import PIL.Image



def test_treerings_inference():
    m = ultralytics.YOLO('ultralytics/ultralytics/cfg/models/26/yolo26-sem.yaml')
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
    m = ultralytics.YOLO('ultralytics/ultralytics/cfg/models/26/yolo26-seg.yaml')
    inputsize = m.args['imgsz']
    module = CellsYOLO_Module(m, px_per_mm=250).eval()
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