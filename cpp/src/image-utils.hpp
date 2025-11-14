#pragma once

#include <expected>

#include "../wasm-big-image/src/png-io.hpp"
#include "../wasm-morpho/src/morphology.hpp"


std::expected<EigenBinaryMap, int> load_and_resize_binary_png(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    const int   dst_width,
    const int   dst_height
);


