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




EigenBinaryMap downsample_2x2(const EigenBinaryMap& x) {
    EigenBinaryMap output(x.dimension(0)/2, x.dimension(1)/2);
    for(int i = 0; i < output.dimension(0); i++)
        for(int j = 0; j < output.dimension(1); j++)
            output(i,j) = (
                (
                    (float) x(i*2,   j*2  ) + 
                    (float) x(i*2+1, j*2  ) + 
                    (float) x(i*2  , j*2+1) + 
                    (float) x(i*2+1, j*2+1)
                ) / 4
            ) > 0;
    return output;
}


// 1GB ~ 16k x 16k rgba
const int MAX_ACCEPTABLE_MEMORY = 1024*1024*1024;

std::expected<EigenBinaryMap, int> load_and_resize_binary_png2(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    int   dst_width,
    int   dst_height
) {
    bool need_to_downsample = false;
    if((dst_width*2) * (dst_height*2) * 4 <= MAX_ACCEPTABLE_MEMORY) {
        dst_width = dst_width*2;
        dst_height = dst_height*2;
        need_to_downsample = true;
    }

    const auto output_x = load_and_resize_binary_png(
        filesize, 
        read_file_callback_p, 
        read_file_handle, 
        dst_width, 
        dst_height
    );
    if(!output_x)
        return output_x;
    EigenBinaryMap output = std::move(output_x.value());

    if(need_to_downsample)
        output = downsample_2x2(output);
    return output;
}

