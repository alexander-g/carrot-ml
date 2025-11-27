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

/** Temporary workaround for poor results with nearest neighbor interpolation */
std::expected<EigenBinaryMap, int> load_and_resize_binary_png2(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    const int   dst_width,
    const int   dst_height
);


// compile time check
template<typename T> uint8_t* to_uint8_p(T* p) {
    static_assert( sizeof(T) == sizeof(uint8_t) );
    return (uint8_t*) p;
}

