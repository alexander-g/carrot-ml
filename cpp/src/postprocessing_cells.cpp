#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <ranges>


#include "./image-utils.hpp"
#include "./postprocessing_cells.hpp"
#include "./utils.hpp"




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


const std::vector<std::array<uint8_t, 3>> COLORS{
    // (255,255,255),
    { 23,190,207},
    {255,127, 14},
    { 44,160, 44},
    {214, 39, 40},
    {148,103,189},
    {140, 86, 75},
    {188,189, 34},
    {227,119,194},
};
const std::array<uint8_t, 3> GRAY{224, 224, 224};

std::optional<EigenRGBMap> colorize_ringmap(
    const CCResult& cc,
    const std::vector<CellInfo>& cell_info,
    const ImageShape& shape
) {
    EigenRGBMap output(shape.first, shape.second, 3);
    output.setZero();
    if(cc.dfs_results.size() != cell_info.size())
        return std::nullopt;

    for(int i = 0; i < cc.dfs_results.size(); i++) {
        const DFS_Result& dfs = cc.dfs_results[i];
        const CellInfo& info  = cell_info[i];
        
        const auto rgb = 
            (info.year_index < 0)
            ? GRAY
            : COLORS[info.year_index % COLORS.size()];

        for(const Index2D& index: dfs.visited) {
            output(index.i, index.j, 0) = rgb[0];
            output(index.i, index.j, 1) = rgb[1];
            output(index.i, index.j, 2) = rgb[2];
        }
    }
    return output;
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

    const CCResult cc_cells = connected_components(mask);
    printf("TODO: remove small objects\n"); fflush(stdout);
    const ImageShape shape{mask.dimension(0), mask.dimension(1)};
    const EigenRGBMap instancemap_rgb = colorize_instancemap(cc_cells, shape);

    const std::expected<Buffer_p, int> instancemap_workshape_png_x = 
        png_compress_image(
            instancemap_rgb.data(), 
            /*width=*/    instancemap_rgb.dimension(1),
            /*height=*/   instancemap_rgb.dimension(0),
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
        /*instancemap_workshape_png = */ instancemap_workshape_png_x.value(),
        /*cells                     = */ std::move(cc_cells)
    };
}




Path polygon_from_treeringspaths(const PathPair& treering) {
    return concat_copy(
        treering.first,
        std::views::reverse(treering.second)
    );
}

std::vector<Path> polygons_from_treeringspaths(const PairedPaths& treering_paths) {
    std::vector<Path> output;
    output.reserve(treering_paths.size());

    for(const PathPair& treering: treering_paths)
        output.push_back(
            concat_copy(
                treering.first,
                std::views::reverse(treering.second)
            )
        );
    
    return output;
}

std::optional<Box> box_from_dfs(const DFS_Result& dfs) {
    if(dfs.visited.empty())
        return std::nullopt;

    Box output{ INFINITY, INFINITY, -INFINITY, -INFINITY };
    for(const Index2D& index: dfs.visited){
        output.x0 = std::min(output.x0, (double)index.j);
        output.y0 = std::min(output.y0, (double)index.i);
        output.x1 = std::max(output.x1, (double)index.j);
        output.y1 = std::max(output.y1, (double)index.i);
    }
    return output;
}


Points indices_to_points(const Indices2D& indices){
    Points points;
    for(const Index2D& index: indices)
        // yx to xy
        points.push_back({ (double)index.j, (double)index.i });
    return points;
}

std::optional<double> estimate_position_within_treering(
    const Points&   cellpoints, 
    const PathPair& treering
) {
    if(treering.first.empty() || treering.second.empty() || cellpoints.empty())
        return std::nullopt;

    return std::nullopt;
    const Point centroid = *average_points(cellpoints);

    const double closest0 = *closest_distance(treering.first, centroid);
    const double closest1 = *closest_distance(treering.second, centroid);
    const double sum = closest0 + closest1;
    if(sum <= 0)
        return std::nullopt;

    return closest0 / sum;
}


int count_positive(const std::vector<bool>& x) {
    int count = 0;
    for(const bool i: x)
        count += i;
    return count;
}


std::expected<CombinedPostprocessingResult, std::string> postprocess_combined(
    const PairedPaths& treering_paths,
    const CCResult&    cells,
    const ImageShape&  workshape,
    const ImageShape&  og_shape
) {
    const int ncells = cells.dfs_results.size();
    const std::vector<Path> treering_polygons = 
        polygons_from_treeringspaths(treering_paths);

    std::vector<CellInfo> cell_info;
    cell_info.reserve(ncells);

    for(int i = 0; i < ncells; i++){
        const DFS_Result& dfs = cells.dfs_results[i];
        const Points cellpoints = scale_points(
            indices_to_points(dfs.visited),
            workshape,
            og_shape
        );
        const int npixels = cellpoints.size();
        if(npixels == 0) // should not happen
            continue;

        int remaining = npixels;

        int largest_overlap_px = 0;
        int largest_overlapping_treering = -1;

        for(int j = 0; j < treering_polygons.size(); j++){
            const Path& treering_polygon = treering_polygons[j];
            const std::vector<bool> points_inside = 
                points_in_polygon(cellpoints, treering_polygon);
            const int n_inside = count_positive(points_inside);

            if(n_inside > largest_overlap_px) {
                largest_overlap_px = n_inside;
                largest_overlapping_treering = j;
            }

            remaining -= n_inside;
            if(remaining <= largest_overlap_px)
                break;
        }
        if(remaining > largest_overlap_px){
            // all treerings processed but only a minority of cell pixels overlap
            // with a ring, this means its mostly outside / indeterminate
            largest_overlap_px = remaining;
            largest_overlapping_treering = -1;
        }

        const double position_within = 
            (largest_overlapping_treering >= 0)
            ? estimate_position_within_treering(
                cellpoints, 
                treering_paths[largest_overlapping_treering]
              ).value_or(-1.0)
            : -1.0;

        cell_info.push_back(CellInfo{
            .id         = i,
            .box_xy     = *box_from_dfs(dfs),
            .year_index = largest_overlapping_treering,
            .area_px    = (double)npixels,
            .position_within = position_within
        });
    }

    const auto expect_ringmap = colorize_ringmap(cells, cell_info, workshape);
    if(!expect_ringmap)
        return std::unexpected("Unexpected error");
    const std::expected<Buffer_p, int> expect_ringmap_workshape_png = 
        png_compress_image(
            expect_ringmap->data(), 
            /*width=*/    expect_ringmap->dimension(1),
            /*height=*/   expect_ringmap->dimension(0),
            /*channels=*/ 3
        );
    if(!expect_ringmap_workshape_png)
        return std::unexpected("Failed to compress instancemap to png");

    return CombinedPostprocessingResult{cell_info, *expect_ringmap_workshape_png};
}




