#include <cstdint>
#include <unordered_map>

#include "./src/postprocessing.hpp"
#include "./src/postprocessing_cells.hpp"
#include "./src/wasm-utils.hpp"



enum Errors_WASM { 
    OK = 0,

    POSTPROCESSING_TREERINGMAPFILE_FAILED = -101,
    POSTPROCESSING_CELLMAPFILE_FAILED     = -102,

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

    const std::string ring_points_json = paired_paths_to_json(output_x->ring_points_xy);

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




void free_output(void* ptr) {
    if(wasm_output_storage.contains(ptr))
        wasm_output_storage.erase(ptr);
    else
        printf("ERROR: tried to remove non-held WASM output\n");
}





}  // extern "C"

