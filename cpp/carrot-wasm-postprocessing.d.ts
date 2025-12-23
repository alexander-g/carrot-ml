

export type Point       = [number, number];
export type Path        = Point[];
export type PathPair    = [Path, Path];
export type PairedPaths = PathPair[];
export type ImageSize   = {width: number, height:number};


export type TreeringPostprocessingResult = {
    /** Binary map with treerings of the size that was used for processing, as PNG */
    treeringmap_workshape_png: File;

    /** Binary map with treerings of the size of original input image, as PNG */
    treeringmap_og_shape_png:  File|null;

    ring_points_xy: PairedPaths;

    /** For internal type checking to differentiate from other result types */
    _type: "treerings";
}

export type CellsPostprocessingResult = {
    /** Binary map with cells of the size that was used for processing, as PNG */
    cellmap_workshape_png: File;

    /** Binary map with cells of the size of original input image, as PNG */
    cellmap_og_shape_png: File|null;

    /** Map with cells colored individually, as PNG */
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

export type CombinedPostprocessingResult = 
    Omit<TreeringPostprocessingResult, '_type'> 
    & Omit<CellsPostprocessingResult,  '_type'> 
    & {

    /** Cells colored by treering, as PNG */
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
