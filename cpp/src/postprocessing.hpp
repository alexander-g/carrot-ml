#ifndef POSTPROCESSING_HPP
#define POSTPROCESSING_HPP

#include <expected>
#include <optional>
#include <vector>

#include "./geometry.hpp"
#include "../wasm-morpho/src/morphology.hpp"
#include "../wasm-big-image/src/util.hpp"




Paths merge_paths(
    const Paths&      paths, 
    const ImageShape& imageshape,
    double max_distance = 0.30, 
    int    ray_width    = 50, 
    double max_overlap  = 0.3, 
    double min_length   = 0.05
);


/** Group tree ring boundaries into tuples */
std::vector<std::pair<int,int>> associate_boundaries(const Paths& paths);

/** Group points from path 0 to corresponding points from path 1 */
std::pair<Path, Path> associate_pathpoints(const Path& path0, const Path& path1);

/** Skeletonize and vectorize boundaries  */
Paths segmentation_to_paths(
    const EigenBinaryMap& mask, 
    double min_length
);

/** Remove paths/parts of paths outside the specified area of interest */
PairedPaths crop_paired_paths_to_aoi(
    const PairedPaths& paths, 
    const AreaOfInterestRect& aoi
);



struct TreeringsPostprocessingResult {
    /** Treeringmap in the size that was used to process it. Encoded as PNG. */
    Buffer_p treeringmap_workshape_png;
    
    /** Treeringmap in the size of the original input. If nullopt, resize manually. */
    std::optional<Buffer_p> treeringmap_og_shape_png;

    // scaled to og_shape
    PairedPaths ring_points_xy;

    /** Same ring points but only those inside the provided area of interest */
    PairedPaths ring_points_in_aoi_xy;
};

std::expected<TreeringsPostprocessingResult, std::string> 
postprocess_treeringmapfile(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    // shape: height first, width second
    const ImageShape& workshape,
    const ImageShape& og_shape,
    const std::optional<AreaOfInterestRect> aoi = std::nullopt,
    // flag to skip resizing mask, takes too long in the browser
    bool do_not_resize_to_og_shape = false
);


#endif