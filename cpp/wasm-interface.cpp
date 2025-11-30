#include <cstdint>
#include <unordered_map>

#include "./src/postprocessing.hpp"
#include "./src/postprocessing_cells.hpp"
#include "./src/wasm-utils.hpp"



enum Errors_WASM { 
    OK = 0,

    POSTPROCESSING_TREERINGMAPFILE_FAILED = -101,
    POSTPROCESSING_CELLMAPFILE_FAILED     = -102,
    POSTPROCESSING_COMBINED_FAILED        = -103,

    UNEXPECTED      = -998,
    NOT_IMPLEMENTED = -999,
};



/** Set of outputs to be held until JS processes them. 
    Stored inside of function closures. */
std::unordered_map<void*, std::function<void()> > wasm_output_storage;



extern "C" {

int postprocess_treeringmapfile_wasm(
    uint32_t    filesize,
    const void* read_file_callback_p,
    const void* read_file_callback_handle,
    uint32_t    workshape_width,
    uint32_t    workshape_height,
    uint32_t    og_shape_width,
    uint32_t    og_shape_height,
    // outputs
    uint8_t**   treeringmap_workshape_png_pp,
    uint32_t*   treeringmap_workshape_png_size_p,
    uint8_t**   ring_points_xy_json_pp,
    uint32_t*   ring_points_xy_json_size_p,
    // returncode because of wasm issues, required
    int* returncode
) {
    *returncode = UNEXPECTED;

    const auto output_x = postprocess_treeringmapfile(
        filesize, 
        read_file_callback_p, 
        read_file_callback_handle, 
        {workshape_height, workshape_width},
        {og_shape_height,  og_shape_width}
    );
    if(!output_x){
        *returncode = POSTPROCESSING_TREERINGMAPFILE_FAILED;
        return *returncode;
    }

    *treeringmap_workshape_png_pp     = output_x->treeringmap_workshape_png->data;
    *treeringmap_workshape_png_size_p = output_x->treeringmap_workshape_png->size;

    // NOTE-2-self: must be non-const for std::move to work
    std::string ring_points_json = paired_paths_to_json(output_x->ring_points_xy);

    *ring_points_xy_json_pp = (uint8_t*)ring_points_json.c_str();
    *ring_points_xy_json_size_p = ring_points_json.size();
    // shouldnt treeringmap_workshape_png be stored too???
    wasm_output_storage.emplace(
        (void*)ring_points_xy_json_pp, 
        [x = std::move(ring_points_json)]() mutable { /* no-op */ } 
    );

    *returncode = OK;
    return *returncode;
}

int postprocess_cellmapfile_wasm(
    uint32_t    filesize,
    const void* read_file_callback_p,
    const void* read_file_callback_handle,
    uint32_t    workshape_width,
    uint32_t    workshape_height,
    uint32_t    og_shape_width,
    uint32_t    og_shape_height,
    // outputs
    uint8_t**   cellmap_workshape_png_pp,
    uint32_t*   cellmap_workshape_png_size_p,
    uint8_t**   instancemap_workshape_png_pp,
    uint32_t*   instancemap_workshape_png_size_p,
    // returncode because of wasm issues, required
    int* returncode
) {
    *returncode = UNEXPECTED;
    const auto expect_output = postprocess_cellmapfile(
        filesize, 
        read_file_callback_p, 
        read_file_callback_handle, 
        {workshape_height, workshape_width},
        {og_shape_height,  og_shape_width}
    );
    if(!expect_output){
        *returncode = POSTPROCESSING_CELLMAPFILE_FAILED;
        return *returncode;
    }
    // shared pointers
    const Buffer_p& cellmap_png = expect_output->cellmap_workshape_png;
    const Buffer_p& instanacemap_png = expect_output->instancemap_workshape_png;

    *cellmap_workshape_png_pp     = cellmap_png->data;
    *cellmap_workshape_png_size_p = cellmap_png->size;
    *instancemap_workshape_png_pp     = instanacemap_png->data;
    *instancemap_workshape_png_size_p = instanacemap_png->size;

    wasm_output_storage.emplace(
        (void*)cellmap_workshape_png_pp, 
        [x = std::move(cellmap_png)]() mutable { /* no-op */ } 
    );
    wasm_output_storage.emplace(
        (void*)instancemap_workshape_png_pp, 
        [x = std::move(instanacemap_png)]() mutable { /* no-op */ } 
    );

    *returncode = OK;
    return *returncode;
}


int postprocess_combined_wasm(
    uint32_t    cellmap_filesize,
    const void* cellmap_read_file_callback_p,
    const void* cellmap_read_file_callback_handle,
    uint32_t    treeringmap_filesize,
    const void* treeringmap_read_file_callback_p,
    const void* treeringmap_read_file_callback_handle,
    uint32_t    workshape_width,
    uint32_t    workshape_height,
    uint32_t    og_shape_width,
    uint32_t    og_shape_height,
    // outputs
    uint8_t**   cellmap_workshape_png_pp,
    uint32_t*   cellmap_workshape_png_size_p,
    uint8_t**   instancemap_workshape_png_pp,
    uint32_t*   instancemap_workshape_png_size_p,
    uint8_t**   treeringmap_workshape_png_pp,
    uint32_t*   treeringmap_workshape_png_size_p,
    uint8_t**   ring_points_xy_json_pp,
    uint32_t*   ring_points_xy_json_size_p,
    uint8_t**   ringmap_workshape_png_pp,
    uint32_t*   ringmap_workshape_png_size_p,
    // returncode because of wasm issues, required
    int* returncode
) {
    *returncode = UNEXPECTED;

    const auto expect_output_cells = postprocess_cellmapfile(
        cellmap_filesize, 
        cellmap_read_file_callback_p, 
        cellmap_read_file_callback_handle, 
        {workshape_height, workshape_width},
        {og_shape_height,  og_shape_width}
    );
    if(!expect_output_cells){
        *returncode = POSTPROCESSING_CELLMAPFILE_FAILED;
        return *returncode;
    }
    const CellsPostprocessingResult& output_cells = *expect_output_cells;

    const auto expect_output_rings = postprocess_treeringmapfile(
        treeringmap_filesize, 
        treeringmap_read_file_callback_p, 
        treeringmap_read_file_callback_handle, 
        {workshape_height, workshape_width},
        {og_shape_height,  og_shape_width}
    );
    if(!expect_output_rings){
        *returncode = POSTPROCESSING_TREERINGMAPFILE_FAILED;
        return *returncode;
    }
    const TreeringsPostprocessingResult& output_rings = *expect_output_rings;

    const auto expect_output_combined = postprocess_combined(
        output_rings.ring_points_xy, 
        output_cells.cells, 
        {workshape_height, workshape_width},
        {og_shape_height,  og_shape_width}
    );
    if(!expect_output_combined){
        *returncode = POSTPROCESSING_COMBINED_FAILED;
        return *returncode;
    }
    const CombinedPostprocessingResult& output_combined = *expect_output_combined;



    // shared pointers
    const Buffer_p& cellmap_png = output_cells.cellmap_workshape_png;
    const Buffer_p& instanacemap_png = output_cells.instancemap_workshape_png;

    *cellmap_workshape_png_pp     = cellmap_png->data;
    *cellmap_workshape_png_size_p = cellmap_png->size;
    *instancemap_workshape_png_pp     = instanacemap_png->data;
    *instancemap_workshape_png_size_p = instanacemap_png->size;

    wasm_output_storage.emplace(
        (void*)cellmap_workshape_png_pp, 
        [x = std::move(cellmap_png)]() mutable { /* no-op */ } 
    );
    wasm_output_storage.emplace(
        (void*)instancemap_workshape_png_pp, 
        [x = std::move(instanacemap_png)]() mutable { /* no-op */ } 
    );

    // shared pointer
    const Buffer_p& treeringmap_png = output_rings.treeringmap_workshape_png;
    *treeringmap_workshape_png_pp     = treeringmap_png->data;
    *treeringmap_workshape_png_size_p = treeringmap_png->size;

    // NOTE 2 self: must be non-const for std::move to work
    std::string ring_points_json = paired_paths_to_json(output_rings.ring_points_xy);

    *ring_points_xy_json_pp = (uint8_t*)ring_points_json.c_str();
    *ring_points_xy_json_size_p = ring_points_json.size();
    wasm_output_storage.emplace(
        (void*)ring_points_xy_json_pp, 
        [x = std::move(ring_points_json)]() mutable { /* no-op */ } 
    );
    wasm_output_storage.emplace(
        (void*)treeringmap_workshape_png_pp, 
        [x = std::move(treeringmap_png)]() mutable { /* no-op */ } 
    );

    // shared pointer
    const Buffer_p& ringmap_png = output_combined.ringmap_rgb_png;
    *ringmap_workshape_png_pp     = ringmap_png->data;
    *ringmap_workshape_png_size_p = ringmap_png->size;
    wasm_output_storage.emplace(
        (void*)ringmap_workshape_png_pp, 
        [x = std::move(ringmap_png)]() mutable { /* no-op */ } 
    );

    *returncode = OK;
    return *returncode;
}




void free_output(void* ptr) {
    if(wasm_output_storage.contains(ptr))
        wasm_output_storage.erase(ptr);
    else
        printf("ERROR: tried to remove non-held WASM output (%p)\n", ptr);
}





}  // extern "C"

