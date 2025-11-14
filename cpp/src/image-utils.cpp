#include "./image-utils.hpp"



/** Convert HxWx4 RGBA data to a boolean mask. */
EigenBinaryMap
rgba_to_binary(const Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>& rgba) {
    const Eigen::Index H = rgba.dimension(0);
    const Eigen::Index W = rgba.dimension(1);
    Eigen::Tensor<bool, 2, Eigen::RowMajor> mask(H, W);

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const uint8_t r = rgba(y, x, 0);
            const uint8_t g = rgba(y, x, 1);
            const uint8_t b = rgba(y, x, 2);
            mask(y, x) = (r | g | b) != 0;
        }
    return mask;
}

std::expected<EigenBinaryMap, int> load_and_resize_binary_png(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    const int   dst_width,
    const int   dst_height
) {
    int rc;

    int32_t image_width, image_height;
    rc = png_get_size(
        filesize,
        read_file_callback_p,
        read_file_handle,
        &image_width,
        &image_height,
        &rc
    );
    if(rc != 0)
        return std::unexpected(rc);

    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> rgba(dst_height, dst_width, 4);
    rc = png_read_patch(
        filesize,
        read_file_callback_p,
        read_file_handle,
        /*src_x      = */ 0,
        /*src_y      = */ 0,
        /*src_width  = */ image_width,
        /*src_height = */ image_height,
        /*dst_width  = */ dst_width,
        /*dst_height = */ dst_height,
        /*dst_buffer = */ rgba.data(),
        /*dst_bufsiz = */ rgba.size(),
        &rc
    );
    if(rc != 0)
        return std::unexpected(rc);

    return rgba_to_binary(rgba);
}



