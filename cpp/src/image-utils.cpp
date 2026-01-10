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


/** Convert a boolean mask to HxWx4 RGBA data. */
Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>
binary_to_rgba(const EigenBinaryMap& mask) {
    const Eigen::Index H = mask.dimension(0);
    const Eigen::Index W = mask.dimension(1);
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> rgba(H, W, 4);

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const bool value = mask(y, x);
            rgba(y, x, 0) = value * 255;
            rgba(y, x, 1) = value * 255;
            rgba(y, x, 2) = value * 255;
            rgba(y, x, 3) = 255;
        }
    return rgba;
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



/** Scale connected components in RLE format from one image size to another.
    If `to_size` is larger than `from_size`, then the result is reversible via
    `(y + 0.5) / (to_size.height / from_size.height)`. */
ListOfRLEComponents scale_rle_components(
    const ListOfRLEComponents& rle_components, 
    const ImageSize& from_size, 
    const ImageSize& to_size
) {
    const double scale_x = to_size.width  / (double)from_size.width;
    const double scale_y = to_size.height / (double)from_size.height;

    // pre-computing how the values would map from `to_size` back onto `from_size`
    std::vector<uint32_t> x_axis, y_axis;
    x_axis.reserve(to_size.width);
    y_axis.reserve(to_size.height);
    for(int x = 0; x < to_size.width; ++x) {
        uint32_t v = (uint32_t)((x + 0.5) / scale_x);
        if(v >= (uint32_t)from_size.width)
            v = (uint32_t)from_size.width - 1;
        x_axis.push_back(v);
    }
    for(int y = 0; y < to_size.height; ++y) {
        uint32_t v = (uint32_t)((y + 0.5) / scale_y);
        if(v >= (uint32_t)from_size.height)
            v = (uint32_t)from_size.height - 1;
        y_axis.push_back(v);
    }

    const auto y_axis_it0 = y_axis.begin();
    const auto y_axis_it1 = y_axis.end();
    const auto x_axis_it0 = x_axis.begin();
    const auto x_axis_it1 = x_axis.end();


    ListOfRLEComponents output;
    for(const RLEComponent& component: rle_components) {
        RLEComponent output_component;

        for(const RLERun& run: component) {
            const uint32_t run_y  = run.row;
            const uint32_t run_x0 = run.start;
            const uint32_t run_x1 = run_x0 + run.len -1;

            const auto to_y0 = std::lower_bound(y_axis_it0, y_axis_it1, run_y);
            if(to_y0 == y_axis_it1 || *to_y0 != run_y )
                // no value would map from `to_size` back to `from_size`
                continue;
            
            const auto to_x0 = std::lower_bound(x_axis_it0, x_axis_it1, run_x0);
            if(to_x0 == x_axis_it1)
                // completely out of bounds
                continue;
            
            // exclusive:
            const auto to_x1 = std::upper_bound(x_axis_it0, x_axis_it1, run_x1);
            if(to_x0 == to_x1)
                continue;
            const auto to_y1 = std::upper_bound(y_axis_it0, y_axis_it1, run_y);

            for(auto it = to_y0; it < to_y1; it++)
                output_component.push_back( RLERun{
                    .row   = (uint32_t) std::distance(y_axis_it0, it),
                    .start = (uint32_t) std::distance(x_axis_it0, to_x0),
                    .len   = (uint32_t) std::distance(to_x0, to_x1)
                } );

        }
        // sort by rows
        std::ranges::sort(output_component, std::less{}, &RLERun::row);
        output.push_back(output_component);
    }
    return output;
}


std::vector<RLERun> flatten_rle_components(const ListOfRLEComponents& components){
    std::vector<RLERun> output;
    output.reserve(components.size() * 10); //estimate

    for(const RLEComponent& component: components)
        for(const RLERun& run: component)
            output.push_back( run );
    
    // sort by rows
    std::ranges::sort(output, std::less{}, &RLERun::row);
    return output;
}


std::expected<Buffer_p, std::string> rasterize_rle_and_encode_as_png_streaming(
    const std::vector<RLERun>& rle_runs,
    const ImageSize& size
) {
    StreamingPNGEncoder spng(size, /*as_binary = */true);

    EigenRGBAMap image_row( 1, size.width, 1 );
    auto run_it = rle_runs.begin();
    for(int y = 0; y < size.height; y++) {
        image_row.setZero();

        while(run_it != rle_runs.end() && run_it->row == y) {
            const uint32_t x0 = run_it->start;
            const uint32_t n  = run_it->len;
            if(x0 >= size.width)
                return std::unexpected(
                    "Start of RLE run exceeds image width: " 
                    + std::to_string(x0) + " vs " + std::to_string(size.width)
                );
            if(x0 + n > size.width)
                return std::unexpected(
                    "End of RLE run exceeds image width: "
                    + std::to_string(x0 + n) + " vs " + std::to_string(size.width)
                );

            std::memset(image_row.data() + x0, 255, n);
            
            run_it++;
        }

        const auto expect_ok = spng.push_image_data(image_row);
        if(!expect_ok)
            return std::unexpected(expect_ok.error());
    }
    // TODO: check if there are runs remaining beyond image height

    return spng.finalize();
}

