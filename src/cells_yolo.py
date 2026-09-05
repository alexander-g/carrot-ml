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



# assuming 10-250um cell sizes, this results in 10-250px
#HARDCODED_GOOD_RESOLUTION = 1000   # px/mm
HARDCODED_GOOD_RESOLUTION = 500   # px/mm

HARDCODED_MIN_CELLSIZE_UM = 10
HARDCODED_MIN_CELLSIZE_PX = \
    HARDCODED_MIN_CELLSIZE_UM * 1000 // HARDCODED_GOOD_RESOLUTION



HARDCODED_DEFAULT_PATCHSIZE = 800


YOLO26S_SEGMENT_PRETRAINED_WEIGHTS_URL = \
    'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt'






class CellsYOLO_Module(torch.nn.Module):
    def __init__(self, yolo:ultralytics.YOLO, px_per_mm:float):
        super().__init__()
        self.yolomodel = yolo.model
        # NOTE: in list to avoid capture by torch.nn.Module
        self._yolo = [yolo]
        self.inputsize = yolo.model.args['imgsz']
        self.px_per_mm = px_per_mm
    
    def forward(self, x:torch.Tensor):
        assert x.ndim == 4
        B,C,H,W = x.shape
        assert H <= self.inputsize and W <= self.inputsize, [self.inputsize, x.shape]

        # pad to avoid misalignment errors
        x = datalib.pad_to_minimum_size(x, self.inputsize)

        output, _  = self.yolomodel(x)
        boxes      = output[0][:,:,:4]
        confidence = output[0][:,:, 4]
        
        outputs = []
        for i in range(B):
            good        = confidence[i] > 0.25
            mask_coeffs = output[0][i, good, 6:]
            good_boxes  = boxes[i, good]
            good_scores = confidence[i, good]

            keep_indices = \
                torchvision.ops.nms(good_boxes, good_scores, iou_threshold=0.3)
            postnms_boxes = good_boxes[keep_indices]
            mask_coeffs   = mask_coeffs[keep_indices]

            proto = output[1][i]
            masks = ultralytics.utils.ops.process_mask(
                proto, 
                mask_coeffs, 
                postnms_boxes, 
                x.shape[-2:], 
                upsample = True
            )  # NHW

            # remove the padding again
            masks = masks[..., :H, :W]

            instancemap = masks_to_instancemap(masks[:,None], remove_overlaps=True)
            outputs.append({'instances': instancemap})
        return outputs




def extras_for_yolo(exporter:torch.package.PackageExporter):
    default_config_yaml = yaml.dump(ultralytics.utils.DEFAULT_CFG_DICT)
    exporter.save_text("ultralytics.extra", "default.yaml", default_config_yaml)


def read_image_as_binary(path:str):
    x = np.array(PIL.Image.open(path).convert('L'))
    x = (x > 0).astype(int)
    return x


def create_dataset_for_yolo(
    splitfile: str, 
    patchsize: int, 
    px_per_mm: float, 
    outputdir: str
) -> str:
    os.makedirs(outputdir, exist_ok=True)
    dataset = InstanceDataset.from_splitfile(
        splitfile, 
        patchsize        = patchsize, 
        px_per_mm        = px_per_mm,
        target_px_per_mm = HARDCODED_GOOD_RESOLUTION,
        cachedir         = outputdir,
    )
    cachedir = os.path.realpath(dataset.cachedir)

    # make sure files have the same name
    old_an_dir = os.path.join(cachedir, 'an')
    new_an_dir = os.path.join(cachedir, 'an2')
    os.makedirs(new_an_dir, exist_ok=True)
    for inf, anf in dataset.items:
        new_anf = os.path.join(new_an_dir, os.path.basename(inf))
        # and is an instancemap, replace with mask
        anf = os.path.join(old_an_dir, os.path.basename(anf))
        shutil.copy(anf, new_anf)

    # yolo wants the folder to be called "images"
    shutil.copytree(
        os.path.join(cachedir, 'in'), 
        os.path.join(cachedir, 'images/'),
        dirs_exist_ok = True,
    )

    # convert masks to .txt files
    labels_txt_dir = os.path.join(cachedir, 'labels/')
    ultralytics.data.converter.convert_segment_masks_to_yolo_seg(
        masks_dir  = new_an_dir, 
        output_dir = labels_txt_dir,
        classes    = 1,
        imread_fn  = read_image_as_binary,
    )


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



def train_yolo_on_cells(
    dataset_yaml:      str, 
    epochs:            int, 
    inputsize:         int,
    batchsize:         int = 4,
    weightsfile:       tp.Optional[str] = None,
    progress_callback: tp.Optional[tp.Callable[[float], None]] = None,
    verbose:           bool = False,
    outputdir:         str = 'checkpoints/',
):
    assert outputdir is not None, 'outputdir currently required'
    outputdir = os.path.abspath(outputdir)
    os.makedirs(outputdir, exist_ok=True)

    if not verbose:
        ultralytics.utils.set_logging('ultralytics', verbose)

    model_yamlfile = os.path.join( os.path.dirname(dataset_yaml), 'model.yaml' )
    open(model_yamlfile, 'w').write(model_yaml)
    model = ultralytics.YOLO(model_yamlfile)
    if weightsfile is not None:
        model.load(weightsfile)
    
    if progress_callback is not None:
        on_epoch_end = lambda trainer: progress_callback(trainer.epoch / epochs)
        model.add_callback("on_train_epoch_end", on_epoch_end)

    if not verbose:
        model.overrides['plots'] = False
    model.overrides['val'] = False

    run_name = time.strftime("%Y-%m-%d_%Hh%Mm%Ss_cells")
    results = model.train(
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
    )

    # re-creating yolo, because it contains some crap
    best_pt = os.path.join(outputdir, run_name, 'weights', 'best.pt')
    model = ultralytics.YOLO(best_pt)

    module = CellsYOLO_Module(model, px_per_mm=HARDCODED_GOOD_RESOLUTION).eval()
    module.inputsize = inputsize
    carrotmodel = CellsYOLO_CARROT(module)
    return carrotmodel




# NOTE: removed rocket emoji because it causes issues in windows
model_yaml = '''
# Ultralytics  AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLO26-seg instance segmentation model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolo26
# Task docs: https://docs.ultralytics.com/tasks/segment

# Parameters
nc: 80 # number of classes
end2end: True # whether to use end-to-end mode
reg_max: 1 # DFL bins
scales: # model compound scaling constants, i.e. 'model=yolo26n-seg.yaml' will call yolo26-seg.yaml with scale 'n'
  # [depth, width, max_channels]
  # n: [0.50, 0.25, 1024] # summary: 309 layers, 3,126,280 parameters, 3,126,280 gradients, 10.5 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 309 layers, 11,505,800 parameters, 11,505,800 gradients, 37.4 GFLOPs
  # m: [0.50, 1.00, 512] # summary: 329 layers, 27,112,072 parameters, 27,112,072 gradients, 132.5 GFLOPs
  # l: [1.00, 1.00, 512] # summary: 441 layers, 31,515,528 parameters, 31,515,528 gradients, 150.9 GFLOPs
  # x: [1.00, 1.50, 512] # summary: 441 layers, 70,693,800 parameters, 70,693,800 gradients, 337.7 GFLOPs

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

  - [[16, 19, 22], 1, Segment26, [nc, 32, 256]] # Segment26(P3, P4, P5)

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
    dataset_yaml = create_dataset_for_yolo(
        splitfile, 
        patchsize, 
        px_per_mm, 
        outputdir = cachedir
    )

    carrotmodel = train_yolo_on_cells(
        dataset_yaml, 
        epochs, 
        inputsize         = patchsize, 
        weightsfile       = weightsfile, 
        progress_callback = progress_callback,
        verbose           = False,
        outputdir         = cachedir,
    )
    return carrotmodel


    return




if __name__ == '__main__':
    import os
    import glob
    import PIL.Image

    # m = ultralytics.YOLO('/home/superuser/Projects/misc/yolo/ultralytics/runs/2026-07-31_cells/best.pt')
    # m = ultralytics.YOLO('/home/superuser/Projects/misc/yolo/ultralytics/runs/segment/train-29/weights/best.pt')
    m = ultralytics.YOLO('/home/superuser/Projects/misc/yolo/ultralytics/runs/segment/train-29/weights/best.pt')
    inputsize = m.args['imgsz']
    print('inputsize:', inputsize)

    m.model.model[-1].max_det *= 2  #useless
    module = CellsYOLO_Module(m, px_per_mm=HARDCODED_GOOD_RESOLUTION*inputsize/800).eval()
    model  = MaskRCNN_Cells_CARROT(module)

    outputdir = 'DEBUG/2026-08-19_yolo-inference/cells-500pxpermm_4'
    os.makedirs(outputdir, exist_ok=True)
    for f in sorted( glob.glob('data/2026-08-21_bw-cells-500/inputs/*') ):
        print(f)
        output = model.process_image(f, px_per_mm=1000, progress_callback=print, batchsize=4)
        PIL.Image.fromarray(output).save(f'{outputdir}/{os.path.basename(f)}.png')
        print()

    print('done')



