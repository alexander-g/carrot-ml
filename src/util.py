import datetime
import hashlib
import importlib
import json
import os
import subprocess
import sys
import typing as tp
import zipfile

import numpy as np
import onnx
import PIL.Image
import tifffile
import torch


# first height, then width
Shape = tp.Tuple[int,int]
ImageAndOGShape = tp.Tuple[torch.Tensor, Shape]


# NOTE: re-implementing here, to avoid traininglib requirement in carrot
def load_image(imagepath:str, mode:str='RGB') -> torch.Tensor:
    return torch.as_tensor(
        np.array(
            PIL.Image.open(imagepath).convert(mode)
        )
    )

def resize_tensor(
    x:     torch.Tensor, 
    shape: Shape, 
    mode:  str,
    align_corners: tp.Optional[bool] = None,
) -> torch.Tensor:
    assert len(x.shape) in [3,4]

    x0 = x
    if len(x0.shape) == 3:
        x = x[np.newaxis]
    y = torch.nn.functional.interpolate(x, shape, mode=mode, align_corners=align_corners)
    if len(x0.shape) == 3:
        y = y[0]
    return y


def load_and_scale_image(path:str, scale:float = 1.0) -> ImageAndOGShape:
    '''Load and scale an image, if a tiff, patchwise to reduce memory'''
    
    if path.endswith('.tiff') or path.endswith('.tif'):
        return load_and_scale_tiff(path, scale)
    else:
        x   = load_image(path)
        x   = x.permute(2,0,1)
        H,W = x.shape[-2:]
        newshape = ( int(H * scale), int(W * scale) )
        x = resize_tensor(x, newshape, 'bilinear')
        x = x.permute(1,2,0)
        return x, (H,W)

def load_and_scale_tiff(path:str, scale:float, patchsize:int=10240) -> ImageAndOGShape:
    '''Load and scale a tiff image, patchwise to reduce memory'''
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        og_shape = page.shape
    H,W = og_shape[:2]
    C = 3 
    newshape = [ C, int(H * scale), int(W * scale) ]
    result = torch.zeros(newshape, dtype=torch.uint8)

    try:
        imdata = tifffile.memmap(path, mode='r')
    except ValueError:
        # cannot memmap if compressed
        with tifffile.TiffFile(path) as tif:
            imdata = tif.pages[0].asarray()  # type: ignore[assignment]
    if len(imdata.shape) == 2:
        imdata = imdata[..., None] # type: ignore[assignment]

    for i in range(0, H, patchsize):
        for j in range(0, W, patchsize):
            patch = torch.as_tensor(
                imdata[i:i+patchsize, j:j+patchsize]
            ).permute(2,0,1)
            h,w = patch.shape[-2:]

            i_ = int(i * scale)
            j_ = int(j * scale)
            i__ = min(int((i + h) * scale), result.shape[1])
            j__ = min(int((j + w) * scale), result.shape[2])

            newpatchshape = (i__ - i_, j__ - j_)
            patch = resize_tensor(patch, newpatchshape, 'bilinear')

            result[:, i_:i__, j_:j__] = patch[:3]

    result = result.permute(1,2,0)
    return result, (H,W)


def scale_points_xy(points_xy:np.ndarray, from_shape:Shape, to_shape:Shape) -> np.ndarray:
    '''Points in XY format, shapes in width-height format'''
    assert points_xy.ndim == 2 and points_xy.shape[-1] == 2

    scale = np.array([
        to_shape[1] / from_shape[1],
        to_shape[0] / from_shape[0],
    ])
    return points_xy * scale


def scale_points_yx(points_yx:np.ndarray, from_shape:Shape, to_shape:Shape) -> np.ndarray:
    '''Points in YX format, shapes in width-height format'''
    assert points_yx.ndim == 2 and points_yx.shape[-1] == 2
    points_xy = points_yx[:,::-1]
    scaled_xy = scale_points_xy( points_xy, from_shape, to_shape )
    scaled_yx = scaled_xy[:,::-1]
    return scaled_yx



def resolve_px_per_mm(
    cli_arg:    str | float | dict[str, float],
    inputfiles: list[str],
) -> dict[str, float]:
    '''Resolve a px_per_mm argument (number, file path, or dict) per input file.'''
    if isinstance(cli_arg, dict):
        return cli_arg
    try:
        px_per_mm = float(cli_arg)
        return {f: px_per_mm for f in inputfiles}
    except ValueError:
        lines     = open(str(cli_arg)).read().strip().split('\n')
        basenames = [os.path.basename(f) for f in inputfiles]
        per_file: dict[str, float] = {}
        for line in lines:
            filename, px_per_mm_str = line.strip().split(':')
            assert filename in basenames, filename
            filepath = inputfiles[basenames.index(filename)]
            per_file[filepath] = float(px_per_mm_str)

        for f in inputfiles:
            assert f in per_file, f

        return per_file


def read_model_metadata(modelpath: str) -> dict[str, str]:
    '''Read metadata dictionary from a .pt.zip or .onnx model file.'''
    try:
        if modelpath.endswith('.onnx'):
            onnx_model = onnx.load(modelpath, load_external_data=False)
            return {p.key: p.value for p in onnx_model.metadata_props}
        with zipfile.ZipFile(modelpath, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('/meta.json') or name == 'meta.json':
                    return json.loads(zf.read(name).decode('utf-8'))
    except Exception:
        pass
    return {}


def construct_model_metadata(
    modeltype:  str,
    dataset:    str | None = None,
    dirty_time: bool       = False,
) -> dict[str, str]:
    '''Construct metadata dictionary for a trained model.'''
    training_time: str = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    if dirty_time:
        training_time += '-dirty'
    dataset_dir: str = (os.path.dirname(dataset) or '.') if dataset else '.'
    return {
        'training-time':  training_time,
        'repo-commit':    git_version('.') or '',
        'dataset-commit': git_version(dataset_dir) or '' if dataset else '',
        'cli-args':       ' '.join(sys.argv),
        'modeltype':      modeltype,
    }


def get_model_version(modelpath:str) -> str:
    with open(modelpath, "rb") as f:
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    
    pattern = '%Y-%m-%d'
    meta = read_model_metadata(modelpath)
    dates: list[str] = []
    for key in ('training-time', 'repo-commit'):
        val = meta.get(key, '')
        try:
            date = datetime.datetime.strptime(val[:len('YYYY-MM-DD')], pattern)
            dates.append(date.strftime(pattern))
        except Exception:
            pass

    if dates:
        date_str = max(dates)
    else:
        try:
            basename = os.path.basename(modelpath)
            date = datetime.datetime.strptime(basename[:len('YYYY-MM-DD')], pattern)
            date_str = date.strftime(pattern)
        except Exception:
            timestamp = os.path.getmtime(modelpath)
            date      = datetime.datetime.fromtimestamp(timestamp)
            date_str  = date.strftime(pattern)

    return f'{date_str}:{sha256}'


def git(folder:str, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(folder), *args],
        text   = True,
        stderr = subprocess.DEVNULL,
    ).strip()

def git_version(folder: str) -> str|None:
    folder = os.path.realpath(folder)
    try:
        repo_root = git(folder, 'rev-parse', '--show-toplevel')
        if repo_root != folder:
            return None
        
        date = git(folder, "show", "-s", "--format=%cs", "HEAD")
        commit_hash = git(folder, "rev-parse", "HEAD")
        dirty = bool(git(folder, "status", "--porcelain"))

        return f"{date}:{commit_hash}{'-dirty' if dirty else ''}"
    except subprocess.CalledProcessError:
        return None


def verify_reproducibility(args: tp.Any) -> list[str]:
    '''Verify repo, dataset, environment, and cli arguments for reproducibility.'''
    errors: list[str] = []

    repo_version: str | None = git_version('.')
    if repo_version is None or repo_version.endswith('-dirty'):
        errors.append(f'Current repository is dirty or not a git repository: {repo_version}')

    dataset_dir: str = os.path.dirname(args.trainsplit) or '.'
    dataset_version: str | None = git_version(dataset_dir)
    if dataset_version is None or dataset_version.endswith('-dirty'):
        errors.append(f'Dataset repository is dirty or not a git repository: {dataset_version}')

    try:
        frozen: str = open('requirements.frozen.txt').read().strip()
        installed: str = subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'],
            text = True,
        ).strip()
    except Exception as exc:
        errors.append(f'Failed to check pip freeze: {exc}')
        frozen, installed = '', ''

    if frozen and installed != frozen:
        errors.append('pip freeze does not match requirements.frozen.txt')

    for line in installed.splitlines():
        if not line.startswith('-e ') or '#egg=' not in line:
            continue
        pkg: str = line.split('#egg=')[-1].split('&')[0].strip()
        try:
            mod_file: str | None = getattr(importlib.import_module(pkg), '__file__', None)
            if mod_file is None:
                errors.append(f'Editable package {pkg} has no __file__')
                continue
            pkg_repo: str = os.path.dirname(os.path.dirname(os.path.realpath(mod_file)))
            pkg_version: str | None = git_version(pkg_repo)
            if pkg_version is None or pkg_version.endswith('-dirty'):
                errors.append(f'Editable package {pkg} is dirty: {pkg_version}')
        except Exception as exc:
            errors.append(f'Failed to check editable package {pkg}: {exc}')

    if getattr(args, 'reuse_yolo', None) is not None:
        errors.append('--reuse-yolo is provided')
    if getattr(args, 'reuse_maskhead', None) is not None:
        errors.append('--reuse-maskhead is provided')

    return errors


