#include "./geometry.hpp"
#include "./image-utils.hpp"



typedef Eigen::Tensor<int, 2, Eigen::RowMajor> EigenIntMap;



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




template <typename T, int R>
using EigenSlice = Eigen::TensorSlicingOp<
    const Eigen::array<Eigen::Index, 2>,
    const Eigen::array<Eigen::Index, 2>,
    Eigen::Tensor<T, 2, R>
>;

template <typename T, int R>
EigenSlice<T,R> crop_3x3_around_index(
    Eigen::Tensor<T,2,R>& x,   //nonconst
    const Index2D& index
) {
    const auto D0 = x.dimension(0);
    const auto D1 = x.dimension(1);
    const int i0 = (index.i <= 0)? 0 : index.i -1;
    const int j0 = (index.j <= 0)? 0 : index.j -1;
    const int i1 = (index.i+1 >= D0)? D0 : index.i + 2;
    const int j1 = (index.j+1 >= D1)? D1 : index.j + 2;
    const Eigen::array<Eigen::Index, 2> 
        offsets = {(Eigen::Index)i0,    (Eigen::Index)j0},
        extents = {(Eigen::Index)i1-i0, (Eigen::Index)j1-j0};

    return x.slice(offsets, extents);
}


/** Sort RLE-encoded connected components in ascending order by size  */
ListOfRLEComponents sort_rle_components_by_size_ascending(ListOfRLEComponents x){
    std::ranges::sort(x, std::less{}, &rle_component_size);
    return x;
}

template <typename T, int D, int R>
bool contains_other_nonzero(const Eigen::Tensor<T,D,R>& x, T i) {
    const auto size = x.size();
    const auto data = x.data();
    for (std::ptrdiff_t idx = 0; idx < size; idx++)
        if(data[idx] != 0  && data[idx] != i)
            return true;
    return false;
}

/** Scale connected components objects, create masks and make sure they are 
    delineated (do not touch each other). */
MaskAndCC scale_objects_and_create_mask(
    const ListOfRLEComponents& objects,
    const ImageSize& from_size,
    const ImageSize& to_size
) {
    const ImageShape from_shape = {from_size.height, from_size.width};
    const ImageShape to_shape   = {to_size.height,   to_size.width};

    // iterate starting with smallest objects to avoid them getting swallowed
    const auto sorted_objects = sort_rle_components_by_size_ascending(objects);
    const auto scaled_objects = 
        scale_rle_components(sorted_objects, from_size, to_size);

    EigenBinaryMap mask(to_size.height, to_size.width);
    EigenIntMap instancemap(to_size.height, to_size.width);
    mask.setZero();
    instancemap.setZero();

    ListOfIndices2D output_objects;
    for(int object_idx = 0; object_idx < scaled_objects.size(); object_idx++) {
        const int object_label = object_idx + 1;
        const Indices2D& object = 
            rle_component_to_dense(scaled_objects[object_idx]);
        const Points points = indices_to_points(object, /*center_pixel=*/true);

        Indices2D quantized_object;
        for(const Point& p: points){
            // xy to yx
            const Index2D quant_p = {
                .i = (Eigen::Index) std::floor(p[1]),
                .j = (Eigen::Index) std::floor(p[0])
            };
            if( mask(quant_p.i, quant_p.j) == true )
                continue;

            const auto cropslice = crop_3x3_around_index(instancemap, quant_p);
            if( contains_other_nonzero(EigenIntMap(cropslice), object_label) )
                continue;

            instancemap(quant_p.i, quant_p.j) = object_label;
            mask(quant_p.i, quant_p.j) = true;
            quantized_object.push_back(quant_p);
        }
        output_objects.push_back(quantized_object);
    }
    return MaskAndCC{.mask=mask, .objects=output_objects};
}





/** Load a binary png, perform streaming connected components on the original 
    sized image and then resize them to a target size.  */
std::expected<MaskAndCC, std::string> 
load_binary_png_connected_components_and_resize(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    const ImageSize target_size
) {
    StreamingConnectedComponents scc;

    ImageSize og_size = {.width=0, .height=0};
    bool all_ok = true;

    const int rc = png_read_streaming(
        filesize, 
        read_file_callback_p, 
        read_file_handle, 
        [&scc, &og_size, &all_ok](const EigenRGBAMap& rows) {
            const EigenBinaryMap binary_rows = rgba_to_binary(rows);

            for(int i=0; i < binary_rows.dimension(0); i++) {
                const EigenBinaryRow row = row_slice(binary_rows, i).value();
                const auto expect_ok = scc.push_image_row(row);
                if(!expect_ok) {
                    all_ok = false;
                    return -1;
                }
            }
            og_size.height += rows.dimension(0);
            og_size.width   = rows.dimension(1);
            return 0;
        }
    );
    if(rc)
        return std::unexpected("Loading png failed: " + std::to_string(rc));
    if(!all_ok)
        return std::unexpected("Connected components failed");

    const CCResultStreaming cc_result = scc.finalize();
    return scale_objects_and_create_mask(cc_result.components, og_size, target_size);
}




/** Scale connected components in RLE format from one image size to another.
    If `to_size` is larger than `from_size`, then the result is reversible. */
ListOfRLEComponents scale_rle_components(
    const ListOfRLEComponents& rle_components, 
    const ImageSize& from_size, 
    const ImageSize& to_size
) {
    // TODO: if(to_size.width == 0 || to_size.height == 0)
    // TODO: if(from_size.width == 0 || from_size.height == 0)

    const double scale_x = to_size.width  / (double)from_size.width;
    const double scale_y = to_size.height / (double)from_size.height;

    ListOfRLEComponents output;
    for(RLEComponent component: rle_components) {
        coalesce_rle_runs(component);
        if(component.empty()) {
            output.push_back({});
            continue;
        }

        RLEComponent output_component;

        //for(const RLERun& run: component) {
        for(int run_i = 0; run_i < component.size(); run_i++) {
            const RLERun& run = component[run_i];

            if(run.len < 1)
                //should not happen after coalesce above
                continue;

            const uint32_t run_y  = run.row;
            const uint32_t run_x0 = run.start;
            const uint32_t run_x1 = run_x0 + run.len -1;  // inclusive / last x


            const bool upscale_y = (scale_y > 1);
            const bool upscale_x = (scale_x > 1);

            //const auto round_fn_y0 = upscale_y? std::ceil : std::floor;
            //const auto round_fn_x0 = upscale_x? std::ceil : std::floor;
            const auto round_fn_y0 = upscale_y
                ? [](double v){ return std::ceil(v); }
                : [](double v){ return std::floor(v); };
            const auto round_fn_x0 = upscale_x
                ? [](double v){ return std::ceil(v); }
                : [](double v){ return std::floor(v); };

            const uint32_t y0_dst = round_fn_y0( run_y * scale_y );
            const uint32_t y1_dst = 
                upscale_y
                ? std::floor( (run_y + 0.999999999) * scale_y  )
                : std::floor( run_y * scale_y );

            const uint32_t x0_dst = round_fn_x0( run_x0 * scale_x );
            const uint32_t x1_dst = 
                upscale_x
                ? std::floor( (run_x1 + 0.999999999) * scale_x )
                : std::floor( run_x1 * scale_x );


            // loop at least once, to avoid missing runs
            for(uint32_t row_dst = y0_dst; row_dst < y1_dst+1; row_dst++) {
                output_component.push_back( RLERun{
                    .row   = row_dst,
                    .start = x0_dst,
                    .len   = (uint32_t) (x1_dst + 1.0) - x0_dst
                } );
            }
        }
        // make sure there is only one run per row
        coalesce_rle_runs(output_component);
        output.push_back(std::move(output_component));
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

