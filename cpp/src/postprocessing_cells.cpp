#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>


#include "./postprocessing_cells.hpp"
#include "./image-utils.hpp"



typedef Eigen::Tensor<int, 2, Eigen::RowMajor>     EigenIntMap;
typedef Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> EigenRGBMap;



/** Convert HSV to RGB. H in range 0-360, S & V in range 0-100 */
std::array<uint8_t, 3> hsv_to_rgb(float h, float s, float v) {
    h = fmodf(h, 360);
    s = std::max(0.0f, std::min(100.0f, s)) / 100.0;
    v = std::max(0.0f, std::min(100.0f, v)) / 100.0;

    if(s <= 0)
        return {(uint8_t)(v*255), (uint8_t)(v*255), (uint8_t)(v*255)};

    const float h_sector = h / 60.0;
    const int   i = (int)h_sector;          // sector 0..5
    const float f = h_sector - i;
    const float p = v * (1 - s);
    const float q = v * (1 - s * f);
    const float t = v * (1 - s * (1 - f));

    const uint8_t vu = (uint8_t) round(v * 255);
    const uint8_t pu = (uint8_t) round(p * 255);
    const uint8_t qu = (uint8_t) round(q * 255);
    const uint8_t tu = (uint8_t) round(t * 255);

    switch (i) {
        case 0:  return {vu, tu, pu};
        case 1:  return {qu, vu, pu};
        case 2:  return {pu, vu, tu};
        case 3:  return {pu, qu, vu};
        case 4:  return {tu, pu, vu};
        default: return {vu, pu, qu}; // case 5
    }
}


EigenRGBMap colorize_instancemap(
    const CCResult& cc, 
    const ImageShape& shape
) {
    EigenRGBMap output(shape.first, shape.second, 3);
    output.setZero();

    std::srand(std::time({}));
    for(const DFS_Result& dfs: cc.dfs_results){
        const double hue = static_cast<double>(std::rand()) / RAND_MAX * 360;
        const double sat = static_cast<double>(std::rand()) / RAND_MAX *10 +80;
        const double val = static_cast<double>(std::rand()) / RAND_MAX *10 +90;
        const auto rgb = hsv_to_rgb(hue, sat, val);

        for(const Index2D& index: dfs.visited) {
            output(index.i, index.j, 0) = rgb[0];
            output(index.i, index.j, 1) = rgb[1];
            output(index.i, index.j, 2) = rgb[2];
        }
    }
    return output;
}



EigenRGBMap classmap_to_instancemap_rgb(const EigenBinaryMap& mask) {
    const CCResult cc = connected_components(mask);
    printf("TODO: remove small objects\n");

    const ImageShape shape{mask.dimension(0), mask.dimension(1)};
    EigenRGBMap rgbmap = colorize_instancemap(cc, shape);
    return rgbmap;
}


std::expected<CellsPostprocessingResult, std::string> postprocess_cellmapfile(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    const ImageShape& workshape,
    const ImageShape& og_shape
) {
    // if not png: error?

    // XXX: python had this: do we still need this?
    // if workshape != classmap.shape:
    //     instancemap_np, _ = \
    //         scipy.ndimage.label( classmap.numpy(), structure=np.ones([3,3]) )
    //     instancemap = resize_tensor(
    //         torch.as_tensor(instancemap_np)[None].float(),
    //         workshape,
    //         mode='nearest',
    //     )[0]
    //     instancemap = delineate_instancemap(instancemap)
    //     classmap = (instancemap > 0)


    const auto expect_mask = load_and_resize_binary_png(
        filesize, 
        read_file_callback_p, 
        read_file_handle,
        workshape.second,  // width
        workshape.first    // height
    );
    if(!expect_mask)
        return std::unexpected("Failed to load cellmap file");
    
    const EigenBinaryMap& mask = expect_mask.value();


    const EigenRGBMap instancemap = classmap_to_instancemap_rgb(mask);
    const std::expected<Buffer_p, int> instancemap_workshape_png_x = 
        png_compress_image(
            instancemap.data(), 
            /*width=*/    instancemap.dimension(1),
            /*height=*/   instancemap.dimension(0),
            /*channels=*/ 3
        );
    if(!instancemap_workshape_png_x)
        return std::unexpected("Failed to compress instancemap to png");

    const std::expected<Buffer_p, int> cellmap_workshape_png_x = 
        png_compress_image(
            to_uint8_p(mask.data()), 
            /*width=*/    mask.dimension(1),
            /*height=*/   mask.dimension(0),
            /*channels=*/ 1
        );
    if(!cellmap_workshape_png_x)
        return std::unexpected("Failed to compress cellmap to png");
    
    return CellsPostprocessingResult{
        /*cellmap_workshape_png     = */ cellmap_workshape_png_x.value(),
        /*instancemap_workshape_png = */ instancemap_workshape_png_x.value()
    };
}




