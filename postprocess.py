import argparse
import glob
import json
import os
import shutil
import sys

import PIL.Image

sys.path.insert(0,'cpp/build/')
import carrot_postprocessing_ext as postp




def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--treerings', required=True, help='Path to .treerings.png')
    parser.add_argument('--cells', help='Path to .cells.png')
    parser.add_argument(
        '--output', 
        required = True, 
        help     = 'Where to save the output. Will copy masks'
    )
    return parser


def main(args: argparse.Namespace):
    treeringfiles = sorted( 
        glob.glob( os.path.join(args.treerings, '*.treerings.png') ) 
    )

    cellfiles: list[str]|None = None
    if args.cells is not None:
        cellfiles = sorted( 
            glob.glob( os.path.join(args.cells, '*.cells.png') ) 
        )
        assert len(cellfiles) == len(treeringfiles), \
            [len(cellfiles), len(treeringfiles)]

        for f0, f1 in zip(cellfiles, treeringfiles):
            assert os.path.basename(f0).split('.')[0] \
                == os.path.basename(f1).split('.')[0]
    
    os.makedirs(args.output, exist_ok=True)

    for i, treeringfile in enumerate(treeringfiles):
        imshape = PIL.Image.open(treeringfile).size[::-1]

        cellfile = cellfiles[i] if cellfiles is not None else None
        output   = postp.postprocess_combined(
            cellfile, 
            treeringfile, 
            workshape = imshape, 
            og_shape  = imshape
        )

        basename = os.path.basename(treeringfile).removesuffix('.treerings.png')

        ring_points: list[tuple] = output['ring_points_xy']
        ring_points_path = \
            os.path.join(args.output, basename) + '.ring_points.json'
        
        with open(ring_points_path, 'w') as f:
            json.dump([ (a.tolist(), b.tolist()) for a,b in ring_points ], f, indent=2)

        treeringspath = os.path.join(args.output, basename) + '.treerings.png'
        shutil.copy(treeringfile, treeringspath)
        
        if cellfile is not None:
            cellpath = os.path.join(args.output, basename) + '.cells.png'
            shutil.copy(cellfile, cellpath)
        
        if 'ringmap_workshape_png' in output:
            ringmappath = os.path.join(args.output, basename) + '.ringmap.png'
            open(ringmappath, 'wb').write( output['ringmap_workshape_png'] )




if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)

    print('done')