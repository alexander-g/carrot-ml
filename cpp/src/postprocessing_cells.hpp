#pragma once

#include <expected>

#include "./geometry.hpp"
#include "../wasm-morpho/src/morphology.hpp"
#include "../wasm-big-image/src/util.hpp"


struct CellsPostprocessingResult {
    const Buffer_p cellmap_workshape_png;
    // EigenBinaryMap cellmap_ogshape;

    const Buffer_p instancemap_workshape_png;
};


std::expected<CellsPostprocessingResult, std::string> postprocess_cellmapfile(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    const ImageShape& workshape,
    const ImageShape& og_shape
);

