#ifndef POSTPROCESSING_HPP
#define POSTPROCESSING_HPP

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

struct TreeringsPostprocessingResult {
    Buffer_p treeringmap_workshape_png;
    Buffer_p treeringmap_og_shape_png;

    // scaled to og_shape
    PairedPaths ring_points_xy;

};

std::optional<TreeringsPostprocessingResult> postprocess_treeringmapfile(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    const ImageShape& workshape,
    const ImageShape& og_shape
);


#endif