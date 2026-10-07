import glob
import os
import shutil
import time
import typing as tp


import numpy as np
import PIL.Image
import torch
import torchvision
import ultralytics
import yaml

assert hasattr(ultralytics, 'YOLO'), 'Adjust PYTHONPATH to import ultralytics'

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '100000000000'
import cv2
cv2.setNumThreads(0)
cv2.setUseOptimized(True)


from traininglib import datalib, modellib
from traininglib.segmentation.connectedcomponents import _relabel
from traininglib.segmentation import (
    grid_for_patches, 
    paste_patch, 
    get_patch_from_grid,
)
from traininglib import trainingloop
from .cc_celldetection import CC_CellsDataset
from .cc_postprocessing import delineate_instancemap
from .maskrcnn_celldetection import (
    masks_to_instancemap, 
    stitch_and_relabel_instancemaps_from_grid,
    MaskRCNN_Cells_CARROT,
    InstanceDataset,
)
from .util import load_and_scale_image
from .cells_yolo_maskhead import (
    MaskHead, 
    MaskHeadTrainStep,
    convert_boxmasks_into_full_result
)



# assuming 10-250um cell sizes, this results in 10-250px
#HARDCODED_GOOD_RESOLUTION = 1000   # px/mm
HARDCODED_GOOD_RESOLUTION = 750   # px/mm
# HARDCODED_GOOD_RESOLUTION = 500   # px/mm

HARDCODED_MIN_CELLSIZE_UM = 10
HARDCODED_MIN_CELLSIZE_PX = \
    HARDCODED_MIN_CELLSIZE_UM * 1000 // HARDCODED_GOOD_RESOLUTION



HARDCODED_DEFAULT_PATCHSIZE = 800


YOLO26S_SEGMENT_PRETRAINED_WEIGHTS_URL = \
    'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt'





class CellsYOLO_Module(torch.nn.Module):
    def __init__(self, yolo:ultralytics.YOLO, maskhead:MaskHead, px_per_mm:float):
        super().__init__()
        assert yolo.args['task'] == 'detect'
        self.yolomodel = yolo.model
        # NOTE: in list to avoid capture by torch.nn.Module
        self._yolo = [yolo]
        self.inputsize = yolo.model.args['imgsz']
        self.maskhead = maskhead
        self.px_per_mm = px_per_mm
    
    def forward(self, x:torch.Tensor):
        assert x.ndim == 4
        B,C,H,W = x.shape
        assert H <= self.inputsize and W <= self.inputsize, [self.inputsize, x.shape]

        # pad to avoid misalignment errors
        x = datalib.pad_to_minimum_size(x, self.inputsize)

        output, _  = self.yolomodel(x)
        boxes      = output[...,:4]
        confidence = output[..., 4]
        
        outputs = []
        for i in range(B):
            good         = confidence[i] > 0.25
            good_boxes   = boxes[i, good]
            good_scores  = confidence[i, good]
            keep_indices = \
                torchvision.ops.nms(good_boxes, good_scores, iou_threshold=0.3)
            postnms_boxes = good_boxes[keep_indices]

            boxmasks = self.maskhead(x[i][None], [postnms_boxes])[0]

            fullmasks = \
                convert_boxmasks_into_full_result((H,W), postnms_boxes, boxmasks > 0)
            fullmasks = (fullmasks > 0)

            instancemap = masks_to_instancemap(
                fullmasks[:,None], 
                largest_only    = False,   # redundant with the new mask head
                remove_overlaps = True,
            )

            outputs.append({'instances': instancemap})
        return outputs




def extras_for_yolo(exporter:torch.package.PackageExporter):
    default_config_yaml = yaml.dump(ultralytics.utils.DEFAULT_CFG_DICT)
    exporter.save_text("ultralytics.extra", "default.yaml", default_config_yaml)


def read_image_as_binary(path:str):
    x = np.array(PIL.Image.open(path).convert('L'))
    x = (x > 0).astype(int)
    return x


def convert_instance_dataset_for_yolo(dataset:InstanceDataset) -> str:
    cachedir = os.path.realpath(dataset.cachedir)

    # yolo wants the folder to be called "images"
    # keeping the old one to avoid cache issues
    old_input_dir = os.path.join(cachedir, 'in')
    new_input_dir = os.path.join(cachedir, 'images/')
    shutil.copytree(old_input_dir, new_input_dir, dirs_exist_ok=True,)

    # replace the dataset items accordingly
    dataset.items = [
        (os.path.join(new_input_dir, os.path.basename(inf)), anf) 
            for inf, anf in dataset.items
    ]

    # convert instance maps to .txt files
    labels_txt_dir = os.path.join(cachedir, 'labels/')
    os.makedirs(labels_txt_dir, exist_ok=True)
    instancemaps = [anf for _inf,anf in dataset.items]
    txtfiles = convert_instancemaps_to_yolo_det(
        inputfiles = instancemaps, 
        outputdir  = labels_txt_dir, 
        classlabel = 1
    )

    # rename the text files, yolo expects them to have the same name as inputs
    inputfiles = [inf for inf,_ in dataset.items]
    for inf, txtf in zip(inputfiles, txtfiles):
        new_txtf = os.path.splitext(os.path.basename(inf))[0] + '.txt'
        shutil.move(txtf, os.path.join(labels_txt_dir, new_txtf))


    dataset_yaml = dataset_yaml_template.format(rootpath=cachedir)
    dataset_yamlfile = os.path.join(cachedir, 'dataset.yaml')
    open(dataset_yamlfile, 'w').write(dataset_yaml)
    return dataset_yamlfile



dataset_yaml_template = '''
path:  {rootpath}
train: images/
val:   images/

names:
  0: background
  1: lumen
'''

def convert_instancemaps_to_yolo_det(
    inputfiles: tp.List[str], 
    outputdir:  str, 
    classlabel: int,
) -> tp.List[str]:
    '''Convert png files with individual objects represented with a unique integer 
       value to `.txt` files as required by yolo object detection. '''
    outputfiles = []
    for f in inputfiles:
        outputlines = []
        outputpath  = os.path.join(outputdir, os.path.basename(f)+'.txt')

        x = np.array(PIL.Image.open(f).convert('L'))
        h,w = x.shape
        uniques = np.unique(x)
        for value in uniques:
            if value == 0:
                continue
            indices = np.argwhere(x == value)
            # from yx to xy
            indices = indices[:,::-1]

            minima  = indices.min(0)
            maxima  = indices.max(0)
            center  = (minima + maxima) / 2 / (w,h)
            boxsize = (maxima - minima) / (w,h)

            line = f'{classlabel} {center[0]} {center[1]} {boxsize[0]} {boxsize[1]}'
            outputlines.append(line)
        
        txt = '\n'.join(outputlines)
        with open(outputpath, 'w') as outf:
            outf.write(txt)

        outputfiles.append(outputpath)
    return outputfiles









def train_yolo_on_cells(
    splitfile:         str,
    px_per_mm:         float, 
    inputsize:         int,
    epochs:            int,
    batchsize:         int = 4,
    weightsfile:       tp.Optional[str] = None,
    progress_callback: tp.Optional[tp.Callable[[float], None]] = None,
    outputdir:         str = 'checkpoints/',
    cachedir:          str = 'cache/',
    # if provided, do not train yolo, simply reuse the provided weights
    reuse_yolo:        tp.Optional[str] = None,
):
    outputdir = os.path.abspath(outputdir)
    os.makedirs(outputdir, exist_ok=True)

    verbose = (progress_callback is None)
    if not verbose:
        ultralytics.utils.set_logging('ultralytics', verbose)

    dataset = InstanceDataset.from_splitfile(
        splitfile        = splitfile, 
        patchsize        = inputsize, 
        px_per_mm        = px_per_mm,
        target_px_per_mm = HARDCODED_GOOD_RESOLUTION,
        cachedir         = cachedir,
    )
    dataset_yaml = convert_instance_dataset_for_yolo(dataset)

    model_yamlfile = os.path.join( os.path.dirname(dataset_yaml), 'model.yaml' )
    open(model_yamlfile, 'w').write(model_yaml)
    yolo = ultralytics.YOLO(model_yamlfile)
    if weightsfile is not None:
        yolo.load(weightsfile)
    
    if progress_callback is not None:
        on_epoch_end = lambda trainer: progress_callback(trainer.epoch / epochs)
        yolo.add_callback("on_train_epoch_end", on_epoch_end)

    if not verbose:
        yolo.overrides['plots'] = False
    yolo.overrides['val'] = False

    run_name = time.strftime("%Y-%m-%d_%Hh%Mm%Ss_cells")
    if reuse_yolo is None:
        results = yolo.train(
            data    = dataset_yaml, 
            epochs  = epochs, 
            imgsz   = inputsize, 
            amp     = False, 
            flipud  = 0.5, 
            fliplr  = 0.5, 
            degrees = 90, 
            workers = 0, 
            batch   = batchsize, 
            verbose = verbose,
            project = outputdir,
            name    = run_name,
            mask_ratio = 2,
        )

        # re-creating yolo, because it contains some crap
        best_pt = os.path.join(outputdir, run_name, 'weights', 'best.pt')
        yolo = ultralytics.YOLO(best_pt)
    else:
        print(f'Re-using YOLO from {reuse_yolo}')
        yolo = ultralytics.YOLO(reuse_yolo)
        weights_dir  = os.path.join(outputdir, run_name, 'weights')
        weights_path = os.path.join(weights_dir, os.path.basename(reuse_yolo))
        os.makedirs(weights_dir, exist_ok=True)
        shutil.copy(reuse_yolo, weights_path)

    dataset = InstanceDataset.from_splitfile(
        splitfile        = splitfile, 
        patchsize        = 480, 
        px_per_mm        = px_per_mm,
        target_px_per_mm = HARDCODED_GOOD_RESOLUTION,
        cachedir         = cachedir,
    )

    head  = MaskHead()
    step  = MaskHeadTrainStep(head)
    ld:tp.Sequence = datalib.create_dataloader( # type: ignore
        dataset, 
        batch_size = batchsize * 2,  # x2 because model is much smaller
        shuffle    = True,
        loader_type = 'threaded',
    )
    trainingloop.train(step, ld, epochs=epochs, progress_callback=progress_callback, lr=1e-3)

    module = \
        CellsYOLO_Module(yolo, head, px_per_mm=HARDCODED_GOOD_RESOLUTION).eval()
    module.inputsize = inputsize
    carrotmodel = CellsYOLO_CARROT(module)
    carrotpath  = os.path.join(outputdir, f'{run_name}/{run_name}.carrot.pt.zip')
    carrotmodel.save(carrotpath)
    return carrotmodel



# NOTE: removed rocket emoji because it causes issues in windows
model_yaml = '''
# Ultralytics  AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLO26 object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolo26
# Task docs: https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
end2end: True # whether to use end-to-end mode
reg_max: 1 # DFL bins
scales: # model compound scaling constants, i.e. 'model=yolo26n.yaml' will call yolo26.yaml with scale 'n'
  # [depth, width, max_channels]
#   n: [0.50, 0.25, 1024] # summary: 260 layers, 2,572,280 parameters, 2,572,280 gradients, 6.1 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 260 layers, 10,009,784 parameters, 10,009,784 gradients, 22.8 GFLOPs
#   m: [0.50, 1.00, 512] # summary: 280 layers, 21,896,248 parameters, 21,896,248 gradients, 75.4 GFLOPs
#   l: [1.00, 1.00, 512] # summary: 392 layers, 26,299,704 parameters, 26,299,704 gradients, 93.8 GFLOPs
#   x: [1.00, 1.50, 512] # summary: 392 layers, 58,993,368 parameters, 58,993,368 gradients, 209.5 GFLOPs

# YOLO26n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO26n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, True]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 22 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)

'''

class CellsYOLO_CARROT(MaskRCNN_Cells_CARROT):
    # override
    def extra_exports(self, pe:torch.package.PackageExporter):
        return extras_for_yolo(pe)



# TODO: combine with treerings, and with the main training script
def start_training_from_carrot(
    filepairs:         tp.List[tp.Tuple[str,str]],
    cachedir:          str,
    px_per_mm:         float,
    epochs:            tp.Optional[int],
    steps:             tp.Optional[int] = None,
    progress_callback: tp.Optional[tp.Callable[[float], None]] = None,
    weightsfile:       tp.Optional[str] = None,
) -> CellsYOLO_CARROT:
    batchsize = 4
    assert epochs is not None or steps is not None
    if epochs is None:
        epochs = int( np.ceil(steps / (len(filepairs) / batchsize)) )
        epochs = max(epochs, 25)
    splitfile = os.path.join(cachedir, 'dataset.yaml')
    datalib.save_file_tuples(splitfile, filepairs)
    patchsize = HARDCODED_DEFAULT_PATCHSIZE

    carrotmodel = train_yolo_on_cells(
        splitfile         = splitfile, 
        px_per_mm         = px_per_mm,
        epochs            = epochs, 
        inputsize         = patchsize, 
        weightsfile       = weightsfile, 
        progress_callback = progress_callback,
        outputdir         = cachedir,
        cachedir          = cachedir,
    )
    return carrotmodel



