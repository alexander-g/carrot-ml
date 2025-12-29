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
    const ListOfIndices2D& cc, 
    const ImageShape& shape
) {
    EigenRGBMap output(shape.first, shape.second, 3);
    output.setZero();

    std::srand(std::time({}));
    for(const Indices2D& indices: cc){
        const double hue = static_cast<double>(std::rand()) / RAND_MAX * 360;
        const double sat = static_cast<double>(std::rand()) / RAND_MAX *10 +80;
        const double val = static_cast<double>(std::rand()) / RAND_MAX *10 +90;
        const auto rgb = hsv_to_rgb(hue, sat, val);

        for(const Index2D& index: indices) {
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
    const ListOfIndices2D& cc,
    const std::vector<CellInfo>&  cell_info,
    const ImageShape& shape
) {
    EigenRGBMap output(shape.first, shape.second, 3);
    output.setZero();
    if(cc.size() != cell_info.size())
        return std::nullopt;

    for(int i = 0; i < cc.size(); i++) {
        const Indices2D& indices = cc[i];
        const CellInfo& info = cell_info[i];
        
        const auto rgb = 
            (info.year_index < 0)
            ? GRAY
            : COLORS[info.year_index % COLORS.size()];

        for(const Index2D& index: indices) {
            output(index.i, index.j, 0) = rgb[0];
            output(index.i, index.j, 1) = rgb[1];
            output(index.i, index.j, 2) = rgb[2];
        }
    }
    return output;
}


/** Convert pixel indices (y first, x second) to points (x first, y second),
    optionally centering on the pixel (+0.5). */
Points indices_to_points(const Indices2D& indices, bool center_pixel){
    const double offset = 0.5 * center_pixel;
    Points points;
    for(const Index2D& index: indices)
        // yx to xy
        points.push_back({ (double)index.j + offset, (double)index.i + offset });
    return points;
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


template <typename T, int D, int R>
bool contains_other_nonzero(const Eigen::Tensor<T,D,R>& x, T i) {
    const auto size = x.size();
    const auto data = x.data();
    for (std::ptrdiff_t idx = 0; idx < size; idx++)
        if(data[idx] != 0  && data[idx] != i)
            return true;
    return false;
}


struct MaskAndCC {
    EigenBinaryMap mask;
    ListOfIndices2D objects;
};

/** Scale connected components objects, create masks and make sure they are 
    delineated (do not touch each other). */
MaskAndCC scale_objects_and_create_mask(
    const ListOfIndices2D& objects,
    const ImageSize& from_size,
    const ImageSize& to_size
) {
    const ImageShape from_shape = {from_size.height, from_size.width};
    const ImageShape to_shape   = {to_size.height,   to_size.width};

    // iterate starting with smallest objects to avoid them getting swallowed
    const auto sorted_objects = std::views::reverse( sort_by_length(objects) );

    EigenBinaryMap mask(to_size.height, to_size.width);
    EigenIntMap instancemap(to_size.height, to_size.width);
    mask.setZero();
    instancemap.setZero();

    ListOfIndices2D output_objects;
    for(int object_idx = 0; object_idx < sorted_objects.size(); object_idx++) {
        const int object_label = object_idx + 1;
        const Indices2D& object = sorted_objects[object_idx];
        const Points points = indices_to_points(object, /*center_pixel=*/true);
        const Points scaled = scale_points(points, from_shape, to_shape);

        Indices2D quantized_object;
        for(const Point& p: scaled){
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
        return std::unexpected("Loading png failed");
    if(!all_ok)
        return std::unexpected("Connected components failed");

    const CCResultStreaming cc_result = scc.finalize();
    return scale_objects_and_create_mask(cc_result.components, og_size, target_size);
}


ListOfIndices2D ccresult_to_indices(CCResult& result) {
    ListOfIndices2D output;
    for(DFS_Result& dfs: result.dfs_results)
        output.push_back(std::move(dfs.visited));
    return output;
}


std::expected<CellsPostprocessingResult, std::string> postprocess_cellmapfile(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    const ImageShape& workshape,
    const ImageShape& og_shape,
    // flag to skip resizing mask, takes too long in the browser
    bool do_not_resize_to_og_shape
) {
    // if not png: error?

    // non-const for std::move
    auto expect_mask_and_cc = load_binary_png_connected_components_and_resize(
        filesize,
        read_file_callback_p,
        read_file_handle,
        {.width = (uint32_t)workshape.second, .height = (uint32_t)workshape.first}
    );
    if(!expect_mask_and_cc)
        return std::unexpected(expect_mask_and_cc.error());
    const EigenBinaryMap& mask = expect_mask_and_cc->mask;
    ListOfIndices2D& cell_ixs = expect_mask_and_cc->objects;

    printf("TODO: remove small objects\n"); fflush(stdout);

    const ImageShape shape{mask.dimension(0), mask.dimension(1)};
    const EigenRGBMap instancemap_rgb = colorize_instancemap(cell_ixs, shape);

    const std::expected<Buffer_p, int> instancemap_workshape_png_x = 
        png_compress_image(
            instancemap_rgb.data(), 
            /*width=*/    instancemap_rgb.dimension(1),
            /*height=*/   instancemap_rgb.dimension(0),
            /*channels=*/ 3
        );
    if(!instancemap_workshape_png_x)
        return std::unexpected("Failed to compress instancemap to png");

    const std::expected<Buffer_p, int> expect_cellmap_workshape_png = 
        png_compress_image(
            to_uint8_p(mask.data()), 
            /*width=*/    mask.dimension(1),
            /*height=*/   mask.dimension(0),
            /*channels=*/ 1
        );
    if(!expect_cellmap_workshape_png)
        return std::unexpected("Failed to compress cellmap to png");
    const Buffer_p cellmap_workshape_png = expect_cellmap_workshape_png.value();

    std::optional<Buffer_p> cellmap_og_shape_png = std::nullopt;
    if(workshape == og_shape)
        cellmap_og_shape_png = cellmap_workshape_png;
    else if(!do_not_resize_to_og_shape) {
        const std::expected<Buffer_p, int> expect_cellmap_og_shape_png = 
            resize_image_and_encode_as_png(
                binary_to_rgba(mask),
                {.width=(uint32_t)og_shape.second, .height=(uint32_t)og_shape.first}
            );
        if(!expect_cellmap_og_shape_png)
            return std::unexpected("Failed to resize to og shape and compress");
        cellmap_og_shape_png = expect_cellmap_og_shape_png.value();
    } //else dont resize here, takes too long
    
    return CellsPostprocessingResult{
        /*cellmap_workshape_png     = */ cellmap_workshape_png,
        /*cellmap_og_shape_png      = */ cellmap_og_shape_png,
        /*instancemap_workshape_png = */ instancemap_workshape_png_x.value(),
        /*cells                     = */ std::move(cell_ixs)
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

std::optional<Box> box_from_indices(const Indices2D& indices) {
    if(indices.empty())
        return std::nullopt;

    Box output{ INFINITY, INFINITY, -INFINITY, -INFINITY };
    for(const Index2D& index: indices){
        output.x0 = std::min(output.x0, (double)index.j);
        output.y0 = std::min(output.y0, (double)index.i);
        output.x1 = std::max(output.x1, (double)index.j);
        output.y1 = std::max(output.y1, (double)index.i);
    }
    return output;
}


std::optional<double> estimate_position_within_treering(
    const Points&   cellpoints, 
    const PathPair& treering
) {
    if(treering.first.empty() || treering.second.empty() || cellpoints.empty())
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



int find_treering_for_cell(
    const std::vector<Path>& treering_polygons, 
    const Points& cellpoints
) {
    const int npixels = cellpoints.size();
    if(npixels == 0)
        return -1;

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
    return largest_overlapping_treering;
}


std::expected<CombinedPostprocessingResult, std::string> postprocess_combined(
    const PairedPaths& treering_paths,
    const ListOfIndices2D& cells,
    const ImageShape&  workshape,
    const ImageShape&  og_shape
) {
    const int ncells = cells.size();
    const std::vector<Path> treering_polygons = 
        polygons_from_treeringspaths(treering_paths);

    std::vector<CellInfo> cell_info;
    cell_info.reserve(ncells);

    for(int i = 0; i < ncells; i++){
        const Indices2D& cell_indices = cells[i];
        const Points cellpoints = scale_points(
            indices_to_points(cell_indices, /*center_pixel=*/false),
            workshape,
            og_shape
        );
        const int npixels = cellpoints.size();
        //if(npixels == 0) // should not happen
        //    continue;

        const int treering = find_treering_for_cell(treering_polygons, cellpoints);

        const double position_within = 
            (treering >= 0)
            ? estimate_position_within_treering(
                cellpoints, 
                treering_paths[treering]
              ).value_or(-1.0)
            : -1.0;

        cell_info.push_back(CellInfo{
            .id         = i,
            .box_xy     = box_from_indices(cell_indices).value_or(Box{0,0,0,0}),
            .year_index = treering,
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




