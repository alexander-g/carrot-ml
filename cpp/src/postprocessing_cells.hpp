#pragma once

#include <expected>
#include <optional>

#include "./geometry.hpp"
#include "../wasm-morpho/src/morphology.hpp"
#include "../wasm-big-image/src/util.hpp"


struct CellsPostprocessingResult {
    /** Cellmap in the size that was used to process it. Encoded as PNG. */
    Buffer_p cellmap_workshape_png;
    /** Treeringmap in the size of the original input. If nullopt, resize manually. */
    std::optional<Buffer_p> cellmap_og_shape_png;

    /** PNG image with cells individually colored */
    Buffer_p instancemap_workshape_png;

    /** Cell points as detected by connected components */
    CCResult cells;
};


std::expected<CellsPostprocessingResult, std::string> postprocess_cellmapfile(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    const ImageShape& workshape,
    const ImageShape& og_shape,
    // flag to skip resizing mask, takes too long in the browser
    bool do_not_resize_to_og_shape = false
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
    Buffer_p ringmap_rgb_png;
};


std::expected<CombinedPostprocessingResult, std::string> postprocess_combined(
    const PairedPaths& treering_paths,
    const CCResult&    cells,
    const ImageShape&  workshape,
    const ImageShape&  og_shape
);
