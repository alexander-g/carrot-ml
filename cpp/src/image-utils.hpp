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


struct MaskAndCC {
    EigenBinaryMap mask;
    ListOfIndices2D objects;
};

/** Load a binary png, perform streaming connected components on the original 
    sized image and then resize them to a target size.  */
std::expected<MaskAndCC, std::string> 
load_binary_png_connected_components_and_resize(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    const ImageSize target_size
);


// compile time check
template<typename T> uint8_t* to_uint8_p(T* p) {
    static_assert( sizeof(T) == sizeof(uint8_t) );
    return (uint8_t*) p;
}



/** Convert a boolean mask to HxWx4 RGBA data. */
Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> 
binary_to_rgba(const EigenBinaryMap& mask);

/** Convert HxWx4 RGBA data to a boolean mask. */
EigenBinaryMap
rgba_to_binary(const Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>& rgba) ;


/** Scale connected components in RLE format from one image size to another.
    If `to_size` is larger than `from_size`, then the result is reversible. */
ListOfRLEComponents scale_rle_components(
    const ListOfRLEComponents& rle_components, 
    const ImageSize& from_size, 
    const ImageSize& to_size
);

std::expected<Buffer_p, std::string> rasterize_rle_and_encode_as_png_streaming(
    const std::vector<RLERun>& rle_runs,
    const ImageSize& size
);


/** Nested std::vector to flat std::vector */
std::vector<RLERun> flatten_rle_components(const ListOfRLEComponents& components);

