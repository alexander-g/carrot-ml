

export type Point       = [number, number];
export type Path        = Point[];
export type PathPair    = [Path, Path];
export type PairedPaths = PathPair[];
export type ImageSize   = {width: number, height:number};


export type TreeringPostprocessingResult = {
    treeringmap_workshape_png: File;

    ring_points_xy: PairedPaths;
}

export type CellsPostprocessingResult = {
    cellmap_workshape_png: File;
    instancemap_workshape_png: File;
}

export type CombinedPostprocessingResult = {
    cellmap_workshape_png: File;
    instancemap_workshape_png: File;
    
    treeringmap_workshape_png: File;
    ring_points_xy: PairedPaths;

    ringmap_workshape_png: File;
}



export declare class CARROT_Postprocessing {
    private constructor();

    postprocess_treeringmapfile(
        treeringmap: File,
        work_size:   ImageSize,
        og_size:     ImageSize,
    ): Promise<TreeringPostprocessingResult | Error>;

    postprocess_cellmapfile(
        cellmap:     File,
        work_size:   ImageSize,
        og_size:     ImageSize,
    ): Promise<CellsPostprocessingResult | Error>;

    postprocess_combined(
        cellmap:     File,
        treeringmap: File,
        work_size:   ImageSize,
        og_size:     ImageSize,
    ): Promise<CombinedPostprocessingResult | Error>;
}


export declare function initialize(): Promise<CARROT_Postprocessing>;
