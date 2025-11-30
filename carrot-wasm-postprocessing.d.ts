

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


    postprocess_combined(
        cellmap:     File|null,
        treeringmap: File|null,
        work_size:   ImageSize,
        og_size:     ImageSize,
    ): Promise<CombinedPostprocessingResult | CellsPostprocessingResult | TreeringPostprocessingResult | Error>;
}


export declare function initialize(): Promise<CARROT_Postprocessing>;
