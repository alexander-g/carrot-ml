

export type Point       = [number, number];
export type Path        = Point[];
export type PathPair    = [Path, Path];
export type PairedPaths = PathPair[];
export type ImageSize   = {width: number, height:number};


export type TreeringPostprocessingResult = {
    treeringmap_workshape_png: File;

    ring_points_xy: PairedPaths;
}


export declare class CARROT_Postprocessing {
    private constructor();

    postprocess_treeringmapfile(
        treeringmap: File,
        work_size:   ImageSize,
        og_size:     ImageSize,
    ): Promise<TreeringPostprocessingResult | Error>;
}


export declare function initialize(): Promise<CARROT_Postprocessing>;
