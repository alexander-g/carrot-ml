#pragma once

#include <expected>

#include "./geometry.hpp"
#include "../wasm-morpho/src/morphology.hpp"
#include "../wasm-big-image/src/util.hpp"


struct CellsPostprocessingResult {
    const Buffer_p cellmap_workshape_png;
    // EigenBinaryMap cellmap_ogshape;

    /** PNG image with cells individually colored */
    const Buffer_p instancemap_workshape_png;

    /** Cell points as detected by connected components */
    const CCResult cells;
};


std::expected<CellsPostprocessingResult, std::string> postprocess_cellmapfile(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    const ImageShape& workshape,
    const ImageShape& og_shape
);





struct CellInfo {
    int    id;
    Box    box_xy;
    int    year_index;
    double area_px;
    double position_within;

};

struct CombinedPostprocessingResult {
    std::vector<CellInfo> cell_info;

    /** PNG image with cells colored by treering they are in */
    const Buffer_p ringmap_rgb_png;
};


std::expected<CombinedPostprocessingResult, std::string> postprocess_combined(
    const PairedPaths& treering_paths,
    const CCResult&    cells,
    const ImageShape&  workshape,
    const ImageShape&  og_shape
);
