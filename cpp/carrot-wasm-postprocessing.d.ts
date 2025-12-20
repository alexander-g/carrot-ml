

export type Point       = [number, number];
export type Path        = Point[];
export type PathPair    = [Path, Path];
export type PairedPaths = PathPair[];
export type ImageSize   = {width: number, height:number};


export type TreeringPostprocessingResult = {
    treeringmap_workshape_png: File;
    treeringmap_og_shape_png:  File|null;

    ring_points_xy: PairedPaths;

    /** For internal type checking to differentiate from other result types */
    _type: "treerings";
}

export type CellsPostprocessingResult = {
    cellmap_workshape_png: File;
    instancemap_workshape_png: File;

    /** For internal type checking to differentiate from other result types */
    _type: "cells";
}


export type CellInfo = {
    id:              number;
    box_xy:          [number,number,number,number];
    year_index:      number;
    area:            number;
    position_within: number;
}

export type CombinedPostprocessingResult = {
    cellmap_workshape_png: File;
    instancemap_workshape_png: File;
    
    treeringmap_workshape_png: File;
    ring_points_xy: PairedPaths;

    ringmap_workshape_png: File;

    cell_info: CellInfo[];

    /** For internal type checking to differentiate from other result types */
    _type: "combined";
}

// any of the above
export type PostprocessingResult = 
    CombinedPostprocessingResult 
    | TreeringPostprocessingResult 
    | CellsPostprocessingResult;


export declare class CARROT_Postprocessing {
    private constructor();


    postprocess_combined(
        cellmap:     File|null,
        treeringmap: File|null,
        work_size:   ImageSize,
        og_size:     ImageSize,
    ): Promise<PostprocessingResult|Error>;

    /** Resize a binary png file to a specified size. */
    resize_mask(
        mask:      File, 
        work_size: ImageSize, 
        og_size:   ImageSize
    ): Promise<File|Error>;
}


export declare function initialize(): Promise<CARROT_Postprocessing>;
