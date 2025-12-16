

export type Point       = [number, number];
export type Path        = Point[];
export type PathPair    = [Path, Path];
export type PairedPaths = PathPair[];
export type ImageSize   = {width: number, height:number};


export type TreeringPostprocessingResult = {
    treeringmap_workshape_png: File;
    treeringmap_og_shape_png:  File;

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

export type EncodingInProgess = {
    file:         Promise<File|Error>;
    abort_handle: number;
};

export declare class CARROT_Postprocessing {
    private constructor();


    postprocess_combined(
        cellmap:     File|null,
        treeringmap: File|null,
        work_size:   ImageSize,
        og_size:     ImageSize,
    ): Promise<
        CombinedPostprocessingResult 
        | CellsPostprocessingResult 
        | TreeringPostprocessingResult 
        | Error
    >;

    /** Resize a binary png file to a specified size. Operation can be aborted
     *  via `abort_resize()`. */
    resize_mask(mask:File, size:ImageSize): EncodingInProgess;

    /** Abort a previously started resizing operation. */
    abort_resize(abort_handle:number): void;
}


export declare function initialize(): Promise<CARROT_Postprocessing>;
