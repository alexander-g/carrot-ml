#include <cstdint>
#include <unordered_map>

#include "./src/image-utils.hpp"
#include "./src/postprocessing.hpp"
#include "./src/postprocessing_cells.hpp"
#include "./src/wasm-utils.hpp"
#include "./wasm-big-image/src/png-io.hpp"



enum Errors_WASM { 
    OK = 0,

    POSTPROCESSING_TREERINGMAPFILE_FAILED = -101,
    POSTPROCESSING_CELLMAPFILE_FAILED     = -102,
    POSTPROCESSING_COMBINED_FAILED        = -103,
    NEITHER_TREERINGSMAP_NOR_CELLMAP_PROVIDED = -104,

    UNEXPECTED      = -998,
    NOT_IMPLEMENTED = -999,
};



/** Set of outputs to be held until JS processes them. 
    Stored inside of function closures. */
std::unordered_map<void*, std::function<void()> > wasm_output_storage;



extern "C" {


int postprocess_combined_wasm(
    // cellmap file callbacks, can be zero/null
    uint32_t    cellmap_filesize,
    const void* cellmap_read_file_callback_p,
    const void* cellmap_read_file_callback_handle,
    // treering file callbacks, can be zero/null
    uint32_t    treeringmap_filesize,
    const void* treeringmap_read_file_callback_p,
    const void* treeringmap_read_file_callback_handle,
    // workshape and original shape
    uint32_t    workshape_width,
    uint32_t    workshape_height,
    uint32_t    og_shape_width,
    uint32_t    og_shape_height,
    // outputs
    uint8_t**   cellmap_workshape_png_pp,
    uint32_t*   cellmap_workshape_png_size_p,
    uint8_t**   cellmap_og_shape_png_pp,
    uint32_t*   cellmap_og_shape_png_size_p,
    uint8_t**   instancemap_workshape_png_pp,
    uint32_t*   instancemap_workshape_png_size_p,
    uint8_t**   treeringmap_workshape_png_pp,
    uint32_t*   treeringmap_workshape_png_size_p,
    uint8_t**   treeringmap_og_shape_png_pp,
    uint32_t*   treeringmap_og_shape_png_size_p,
    uint8_t**   ring_points_xy_json_pp,
    uint32_t*   ring_points_xy_json_size_p,
    uint8_t**   ringmap_workshape_png_pp,
    uint32_t*   ringmap_workshape_png_size_p,
    uint8_t**   cell_info_json_pp,
    uint32_t*   cell_info_json_size_p,
    // returncode because of wasm issues, required
    int* returncode
) {
    *returncode = UNEXPECTED;

    const bool have_cellmap = 
        (cellmap_filesize > 0) && (cellmap_read_file_callback_p != NULL);
    const bool have_treeringmap = 
        (treeringmap_filesize > 0) && (treeringmap_read_file_callback_p != NULL);
    const bool have_both = have_cellmap && have_treeringmap;
    if(!have_cellmap && !have_treeringmap){
        *returncode = NEITHER_TREERINGSMAP_NOR_CELLMAP_PROVIDED;
        return *returncode;
    }



    std::expected<CellsPostprocessingResult, std::string> expect_output_cells =
        std::unexpected("not initialized");
    if(have_cellmap){
        expect_output_cells = postprocess_cellmapfile(
            cellmap_filesize, 
            cellmap_read_file_callback_p, 
            cellmap_read_file_callback_handle, 
            {workshape_height, workshape_width},
            {og_shape_height,  og_shape_width},
            /* do_not_resize_to_og_shape = */ true
        );
        if(!expect_output_cells){
            *returncode = POSTPROCESSING_CELLMAPFILE_FAILED;
            return *returncode;
        }
    }


    std::optional<TreeringsPostprocessingResult> expect_output_rings = std::nullopt;
    if(have_treeringmap){ 
        expect_output_rings = postprocess_treeringmapfile(
            treeringmap_filesize, 
            treeringmap_read_file_callback_p, 
            treeringmap_read_file_callback_handle, 
            {workshape_height, workshape_width},
            {og_shape_height,  og_shape_width},
            /* do_not_resize_to_og_shape = */ true
        );
        if(!expect_output_rings){
            *returncode = POSTPROCESSING_TREERINGMAPFILE_FAILED;
            return *returncode;
        }
    }



    std::expected<CombinedPostprocessingResult, std::string> expect_output_combined =
        std::unexpected("not initialized");
    if(have_both){
        const CellsPostprocessingResult& output_cells = *expect_output_cells;
        const TreeringsPostprocessingResult& output_rings = *expect_output_rings;
        expect_output_combined = postprocess_combined(
            output_rings.ring_points_xy, 
            output_cells.cells, 
            {workshape_height, workshape_width},
            {og_shape_height,  og_shape_width}
        );
        if(!expect_output_combined){
            *returncode = POSTPROCESSING_COMBINED_FAILED;
            return *returncode;
        }
    }




    if(have_cellmap){
        const CellsPostprocessingResult& output_cells = *expect_output_cells;

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
    }

    
    if(have_treeringmap) {
        const TreeringsPostprocessingResult& output_rings = *expect_output_rings;

        // shared pointer
        const Buffer_p& treeringmap_workshape_png = 
            output_rings.treeringmap_workshape_png;
        *treeringmap_workshape_png_pp     = treeringmap_workshape_png->data;
        *treeringmap_workshape_png_size_p = treeringmap_workshape_png->size;

        // NOTE 2 self: must be non-const for std::move to work
        std::string ring_points_json = 
            paired_paths_to_json(output_rings.ring_points_xy);

        *ring_points_xy_json_pp = (uint8_t*)ring_points_json.c_str();
        *ring_points_xy_json_size_p = ring_points_json.size();
        wasm_output_storage.emplace(
            (void*)ring_points_xy_json_pp, 
            [x = std::move(ring_points_json)]() mutable { /* no-op */ } 
        );
        wasm_output_storage.emplace(
            (void*)treeringmap_workshape_png_pp, 
            [x = std::move(treeringmap_workshape_png)]() mutable { /* no-op */ } 
        );


        if(output_rings.treeringmap_og_shape_png) {
            // shared pointer
            const Buffer_p& treeringmap_og_shape_png = 
                output_rings.treeringmap_og_shape_png.value();
            *treeringmap_og_shape_png_pp     = treeringmap_og_shape_png->data;
            *treeringmap_og_shape_png_size_p = treeringmap_og_shape_png->size;

            wasm_output_storage.emplace(
                (void*)treeringmap_og_shape_png_pp, 
                [x = std::move(treeringmap_og_shape_png)]() mutable { /* no-op */ } 
            );
        } else {
            *treeringmap_og_shape_png_pp     = nullptr;
            *treeringmap_og_shape_png_size_p = 0;
        }
    }

    if(have_both){
        const CombinedPostprocessingResult& output_combined = 
            *expect_output_combined;

        // shared pointer
        const Buffer_p& ringmap_png = output_combined.ringmap_rgb_png;
        *ringmap_workshape_png_pp     = ringmap_png->data;
        *ringmap_workshape_png_size_p = ringmap_png->size;

        // NOTE: must be non-const for std::move to work
        std::string cell_info_json = cell_info_to_json(output_combined.cell_info);

        *cell_info_json_pp = (uint8_t*)cell_info_json.c_str();
        *cell_info_json_size_p = cell_info_json.size();

        
        wasm_output_storage.emplace(
            (void*)cell_info_json_pp, 
            [x = std::move(cell_info_json)]() mutable { /* no-op */ } 
        );
        wasm_output_storage.emplace(
            (void*)ringmap_workshape_png_pp, 
            [x = std::move(ringmap_png)]() mutable { /* no-op */ } 
        );
    }

    *returncode = OK;
    return *returncode;
}


int resize_mask(
    // file read callbacks
    uint32_t    filesize,
    const void* read_file_callback_p,
    const void* read_file_callback_handle,
    // workshape and original/target shape
    uint32_t    workshape_width,
    uint32_t    workshape_height,
    uint32_t    og_shape_width,
    uint32_t    og_shape_height,
    // output
    uint8_t**   png_buffer_pp,
    uint32_t*   png_buffer_size_p,
    // returncode because of wasm issues, required
    int* returncode
) {
    *returncode = UNEXPECTED;

    const auto expect_mask = load_and_resize_binary_png2(
        filesize, 
        read_file_callback_p, 
        read_file_callback_handle,
        workshape_width,
        workshape_height
    );
    if(!expect_mask){
        *returncode = expect_mask.error();
        return *returncode;
    }
    const EigenBinaryMap& mask = expect_mask.value();

    const std::expected<Buffer_p, int> expect_resized_png = 
        resize_image_and_encode_as_png(
            // NOTE: keep as bool, for faster encoding
            mask,
            {.width=og_shape_width, .height=og_shape_height}
        );
    if(!expect_resized_png){
        *returncode = expect_resized_png.error();
        return *returncode;
    }
    const Buffer_p resized_png = expect_resized_png.value();

    *png_buffer_pp     = resized_png->data;
    *png_buffer_size_p = resized_png->size;
    
    wasm_output_storage.emplace(
        (void*)png_buffer_pp, 
        [x = std::move(resized_png)]() mutable { /* no-op */ } 
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

