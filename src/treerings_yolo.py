import shutil
import os
import time
import typing as tp

import numpy as np
import torch
import ultralytics
assert hasattr(ultralytics, 'YOLO'), 'Adjust PYTHONPATH to import ultralytics'

from traininglib import datalib

from .cells_yolo import extras_for_yolo
from .treeringmodel import Treerings_CARROT, TreeringsInference, TreeringsDataset
from .util import load_and_scale_image



# assuming minimum 0.1mm tree ring width, this results in 25px, should be enough
#HARDCODED_GOOD_RESOLUTION = 250   # px/mm
HARDCODED_GOOD_RESOLUTION = 150   # px/mm

def resolution_to_scale(px_per_mm:float) -> float:
    return  HARDCODED_GOOD_RESOLUTION / px_per_mm

HARDCODED_DEFAULT_PATCHSIZE = 640   # px

YOLO26S_SEMANTIC_PRETRAINED_WEIGHTS_URL = \
    'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-sem-ade20k.pt'



class TreeringsYOLO_Module(torch.nn.Module):
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

        output = self.yolomodel(x)
        output = torch.nn.functional.interpolate(output, x.shape[-2:], mode='bilinear')
        
        # remove the padding again
        output = output[..., :H, :W]
        return output[:, -1:]



def create_dataset_for_yolo(
    splitfile: str, 
    patchsize: int, 
    px_per_mm: float, 
    outputdir: str
) -> str:
    dataset = TreeringsDataset.from_splitfile(
        splitfile, 
        patchsize        = patchsize, 
        px_per_mm        = px_per_mm,
        target_px_per_mm = HARDCODED_GOOD_RESOLUTION,
        dilation         = 2,
        cachedir         = outputdir,
    )
    cachedir = os.path.realpath(dataset.cachedir)

    # make sure files have the same name
    old_an_dir = os.path.join(cachedir, 'targets')
    new_an_dir = os.path.join(cachedir, 'masks')
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

    dataset_yaml = dataset_yaml_template.format(rootpath=cachedir)
    dataset_yamlfile = os.path.join(cachedir, 'dataset.yaml')
    open(dataset_yamlfile, 'w').write(dataset_yaml)
    return dataset_yamlfile



dataset_yaml_template = '''
path:  {rootpath}
train: images
val:   images

masks_dir: masks # semantic mask directory

names:
  0: background
  1: treeringboundary

label_mapping:
  -1: ignore_label
  0: 0
  255: 1
'''




def train_yolo_on_treerings(
    dataset_yaml: str, 
    epochs:       int, 
    inputsize:    int, 
    batchsize:    int = 4,
    weightsfile:  tp.Optional[str] = None,
    progress_callback: tp.Optional[tp.Callable[[float], None]] = None,
    verbose:      bool = False,
    outputdir:    str = 'checkpoints/',
):
    assert outputdir is not None, 'outputdir currently required'
    outputdir = os.path.abspath(outputdir)
    os.makedirs(outputdir, exist_ok=True)

    if not verbose:
        ultralytics.utils.set_logging('ultralytics', verbose)
    
    model_yamlfile = os.path.join( os.path.dirname(dataset_yaml), 'model.yaml' )
    open(model_yamlfile, 'w').write(modified_model_yaml)
    model = ultralytics.YOLO(model_yamlfile)
    if weightsfile is not None:
        model.load(weightsfile)
    if progress_callback is not None:
        on_epoch_end = lambda trainer: progress_callback(trainer.epoch / epochs)
        model.add_callback("on_train_epoch_end", on_epoch_end)

    if not verbose:
        model.overrides['plots'] = False
    model.overrides['val'] = False

    run_name = time.strftime("%Y-%m-%d_%Hh%Mm%Ss_treerings")
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
    # required for pickling:
    model.clear_callback('on_train_epoch_end')

    # re-creating yolo, because it contains some crap
    best_pt = os.path.join(outputdir, run_name, 'weights', 'best.pt')
    model = ultralytics.YOLO(best_pt)


    module = TreeringsYOLO_Module(model, px_per_mm=HARDCODED_GOOD_RESOLUTION).eval()
    carrotmodel = \
        TreeringsYOLO_CARROT(TreeringsInference(module, patchsize=inputsize))
    # carrotmodel.save
    return carrotmodel



model_yaml = '''
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLO26 semantic segmentation model with P3/8 and P4/16 head inputs
# Model docs: https://docs.ultralytics.com/models/yolo26
# Task docs: https://docs.ultralytics.com/tasks/semantic

# Parameters
nc: 19 # number of classes (Cityscapes default)
scales: # model compound scaling constants, i.e. 'model=yolo26n-sem.yaml' will call yolo26-sem.yaml with scale 'n'
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

# YOLO26n semantic segmentation head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  - [[16, 13], 1, SemanticSegment, [nc]] # SemanticSegment
'''



modified_model_yaml = '''
nc: 19
scales: # model compound scaling constants, i.e. 'model=yolo26n-sem.yaml' will call yolo26-sem.yaml with scale 'n'
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

# YOLO26n semantic segmentation head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  # extra block for higher resolution
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, True]] # 19 (P2/4)

  - [[-1, 16, 13], 1, SemanticSegment, [nc]] # SemanticSegment
'''




class TreeringsYOLO_CARROT(Treerings_CARROT):
    # override
    def extra_exports(self, pe:torch.package.PackageExporter):
        return extras_for_yolo(pe)




def start_training_from_carrot(
    filepairs:         tp.List[tp.Tuple[str,str]],
    cachedir:          str,
    px_per_mm:         float,
    epochs:            tp.Optional[int],
    steps:             tp.Optional[int] = None,
    progress_callback: tp.Optional[tp.Callable[[float], None]] = None,
    weightsfile:       tp.Optional[str] = None,
) -> Treerings_CARROT:
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

    carrotmodel = train_yolo_on_treerings(
        dataset_yaml, 
        epochs, 
        inputsize         = patchsize, 
        weightsfile       = weightsfile, 
        progress_callback = progress_callback,
        verbose           = False,
        outputdir         = cachedir,
    )
    return carrotmodel





if __name__ == '__main__':
    import os
    import glob
    import PIL.Image

    m = ultralytics.YOLO('ultralytics/runs/semantic/train-16/weights/best.pt')
    inputsize = m.args['imgsz']

    #module = TreeringsYOLO_Module(m, px_per_mm=250*inputsize/1024).eval()
    module = TreeringsYOLO_Module(m, px_per_mm=150).eval()
    model  = Treerings_CARROT(TreeringsInference(module, patchsize=inputsize))
    
    # imf = 'data/2026-06-29_bw-rings+extra-250/inputs/001_250px-per-mm.png'
    # output = model.process_image(imf, px_per_mm=250, progress_callback=print)


    outputdir = 'DEBUG/2026-08-19_yolo-inference/2026-08-23_rings-250pxpermm'
    os.makedirs(outputdir, exist_ok=True)
    for f in glob.glob('data/2026-06-29_bw-rings+extra-250/inputs/*'):
        print(f)
        output = model.process_image(f, px_per_mm=250, progress_callback=print, batchsize=1)
        PIL.Image.fromarray(output).save(f'{outputdir}/{os.path.basename(f)}.png')
        print()

    print('done')



