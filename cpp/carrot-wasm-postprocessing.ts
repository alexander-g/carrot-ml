import type {
    initialize as Iinitialize,
    CARROT_Postprocessing as ICARROT_Postprocessing,
    PostprocessingResult,
    CellInfo,
    PairedPaths,
    PathPair,
    Path,
    Point,
    ImageSize,
} from "./carrot-wasm-postprocessing.d.ts"

import { 
    is_array_of_type,
    is_number_array,
    is_object,
    has_number_property,
    has_property_of_type,
} from "./dep.ts"



// for better readability
type fn_pointer = number;
type pointer = number;


type CARROT_Postprocessing_WASM = {

    _postprocess_combined_wasm: (
        cellmap_filesize:                      number,
        cellmap_read_file_callback_p:          fn_pointer,
        cellmap_read_file_callback_handle:     number,
        treeringmap_filesize:                  number,
        treeringmap_read_file_callback_p:      fn_pointer,
        treeringmap_read_file_callback_handle: number,
        workshape_width:      number,
        workshape_height:     number,
        og_shape_width:       number,
        og_shape_height:      number,
        // outputs
        cellmap_workshape_png:           pointer,
        cellmap_workshape_png_size:      pointer,
        cellmap_og_shape_png:            pointer,
        cellmap_og_shape_png_pp_size:    pointer,
        instancemap_workshape_png:       pointer,
        instancemap_workshape_png_size:  pointer,
        treeringmap_workshape_png:       pointer,
        treeringmap_workshape_png_size:  pointer,
        treeringmap_og_shape_png:        pointer,
        treeringmap_og_shape_png_size:   pointer,
        ring_points_xy_json:             pointer,
        ring_points_xy_json_size:        pointer,
        ringmap_workshape_png_pp:        pointer,
        ringmap_workshape_png_size_p:    pointer,
        cell_info_json_size:             pointer,
        cell_info_json:                  pointer,

        returncode: pointer,
    ) => number,

    _resize_mask: (
        filesize:                  number,
        read_file_callback_p:      pointer,
        read_file_callback_handle: pointer,
        // workshape and original/target shapes
        workshape_width:  number,
        workshape_height: number,
        og_shape_width:   number,
        og_shape_height:  number,
        // output
        png_buffer_pp:     pointer,
        png_buffer_size_p: pointer,
        // returncode because of wasm issues, required
        returncode: pointer
    ) => number,

    _free_output: (p:pointer) => void,


    _malloc: (nbytes:number) => pointer,
    _free:   (ptr:pointer) => void,

    // deno-lint-ignore no-explicit-any
    addFunction: ( fn: (...args: any[]) => unknown, argtypes:string ) => number,
    
    HEAPU8: {
        set: (src:Uint8Array, dst:pointer) => void,
        slice: (start:number, end:number) => Uint8Array<ArrayBuffer>,
        [i:number]: number,
    }
    HEAP32: {
        [i:number]: number,
    },
    HEAP64: {
        [i:number]: bigint,
    },

    Asyncify: {
        // deno-lint-ignore no-explicit-any
        handleAsync: (fn:(...args:any[]) => Promise<unknown>) => unknown,
        currData: number|null,
    },
}


function wait(ms: number): Promise<unknown> {
    return new Promise((resolve) => {
        setTimeout(() => resolve(0), ms)
    })
}



export class CARROT_Postprocessing implements ICARROT_Postprocessing {
    constructor(private wasm:CARROT_Postprocessing_WASM){
        this.#read_file_callback_ptr = 
            wasm.addFunction(this.#read_file_callback, 'iiijj');
    }


    async postprocess_combined(
        cellmap:     File|null,
        treeringmap: File|null,
        work_size:   ImageSize,
        og_size:     ImageSize,
    ): Promise<PostprocessingResult|Error> {

        let cells_handle:number = 0;
        let rings_handle:number = 0;
        if(cellmap){
            cells_handle = this.#handle_counter++;
            this.#read_file_callback_table[cells_handle] = cellmap;
        }
        if(treeringmap){
            rings_handle = this.#handle_counter++;
            this.#read_file_callback_table[rings_handle] = treeringmap;
        }


        const rc_ptr:pointer = this.#malloc(4, /*fill=*/255)

        const cellmap_workshape_png_pp:pointer         = this.#malloc(8);
        const cellmap_workshape_png_size_p:pointer     = this.#malloc(8);
        const cellmap_og_shape_png_pp:pointer         = this.#malloc(8);
        const cellmap_og_shape_png_size_p:pointer     = this.#malloc(8);
        const instancemap_workshape_png_pp:pointer     = this.#malloc(8);
        const instancemap_workshape_png_size_p:pointer = this.#malloc(8);
        const treeringmap_workshape_png_pp:pointer     = this.#malloc(8);
        const treeringmap_workshape_png_size_p:pointer = this.#malloc(8);
        const treeringmap_og_shape_png_pp:pointer      = this.#malloc(8);
        const treeringmap_og_shape_png_size_p:pointer  = this.#malloc(8);
        const ring_points_xy_json_pp:pointer           = this.#malloc(8);
        const ring_points_xy_json_size_p:pointer       = this.#malloc(8);
        const ringmap_workshape_png_pp:pointer         = this.#malloc(8);
        const ringmap_workshape_png_size_p:pointer     = this.#malloc(8);
        const cell_info_json_pp:pointer                = this.#malloc(8);
        const cell_info_json_size_p:pointer            = this.#malloc(8);

        try {
            let rc:number = await this.wasm._postprocess_combined_wasm(
                cellmap?.size ?? 0, 
                this.#read_file_callback_ptr, 
                cells_handle, 
                treeringmap?.size ?? 0, 
                this.#read_file_callback_ptr, 
                rings_handle, 

                work_size.width, 
                work_size.height,
                og_size.width,
                og_size.height,

                cellmap_workshape_png_pp,
                cellmap_workshape_png_size_p,
                cellmap_og_shape_png_pp,
                cellmap_og_shape_png_size_p,
                instancemap_workshape_png_pp,
                instancemap_workshape_png_size_p,
                treeringmap_workshape_png_pp,
                treeringmap_workshape_png_size_p,
                treeringmap_og_shape_png_pp,
                treeringmap_og_shape_png_size_p,
                ring_points_xy_json_pp,
                ring_points_xy_json_size_p,
                ringmap_workshape_png_pp,
                ringmap_workshape_png_size_p,
                cell_info_json_pp,
                cell_info_json_size_p,

                rc_ptr
            )
            // NOTE: the wasm function above returns before the execution is 
            // finished because of async issues, so currently polling until done
            while(this.wasm.Asyncify.currData != null)
                await wait(1);
    
            rc = (rc == 0)? this.wasm.HEAP32[rc_ptr >> 2]! : rc;
            if(rc != 0)
                return new Error(`WASM error code = ${rc}`)

            let cellmap_workshape:File|null = null;
            let cellmap_og_shape: File|null = null;
            let instancemap_workshape:File|null = null;
            if(cellmap) {
                const cellmap_workshape_png_u8:Uint8Array<ArrayBuffer>|null = 
                    this.#read_dynamic_output_buffer(
                        cellmap_workshape_png_pp, 
                        cellmap_workshape_png_size_p
                    )
                if(cellmap_workshape_png_u8 != null)
                    cellmap_workshape = 
                        new File([cellmap_workshape_png_u8], 'cellmap.png')
                
                const cellmap_og_shape_u8:Uint8Array<ArrayBuffer>|null = 
                    this.#read_dynamic_output_buffer(
                        cellmap_og_shape_png_pp,
                        cellmap_og_shape_png_size_p
                    )
                if(cellmap_og_shape_u8 != null)
                    cellmap_og_shape = new File([cellmap_og_shape_u8], 'cellmap.png')

                const instancemap_workshape_png_u8:Uint8Array<ArrayBuffer>|null 
                    = this.#read_dynamic_output_buffer(
                        instancemap_workshape_png_pp,
                        instancemap_workshape_png_size_p
                    )
                if(instancemap_workshape_png_u8 != null)
                    instancemap_workshape = 
                        new File([instancemap_workshape_png_u8], 'instancemap.png')
            }


            let treeringmap_workshape_shape_png: File|null = null;
            let treeringmap_og_shape_png:        File|null = null;
            let paired_paths:PairedPaths|null = null;
            if(treeringmap) {
                const treeringmap_workshape_png_u8:Uint8Array<ArrayBuffer>|null 
                    = this.#read_dynamic_output_buffer(
                        treeringmap_workshape_png_pp,
                        treeringmap_workshape_png_size_p
                    )
                if(treeringmap_workshape_png_u8 != null)
                    treeringmap_workshape_shape_png = 
                        new File([treeringmap_workshape_png_u8], 'treeringmap.png')
                
                const treeringmap_og_shape_png_u8:Uint8Array<ArrayBuffer>|null 
                    = this.#read_dynamic_output_buffer(
                        treeringmap_og_shape_png_pp,
                        treeringmap_og_shape_png_size_p
                    )
                if(treeringmap_og_shape_png_u8 != null)
                    treeringmap_og_shape_png = 
                        new File([treeringmap_og_shape_png_u8], 'treeringmap.png')

                const ring_points_xy_json_u8:Uint8Array|null = 
                    this.#read_dynamic_output_buffer(
                        ring_points_xy_json_pp,
                        ring_points_xy_json_size_p
                    )
                if(ring_points_xy_json_u8 == null)
                    return new Error('WASM error: did not return ring points')
                const obj:unknown = 
                    JSON.parse(new TextDecoder().decode(ring_points_xy_json_u8));
                paired_paths = validate_paired_paths(obj)
                if(paired_paths == null)
                    return new Error('WASM-JS communication inconcistencies')
            }


            let ringmap_workshape_png_u8:Uint8Array<ArrayBuffer>|null = null;
            let cell_info:CellInfo[]|null = null;
            if(cellmap && treeringmap) {
                ringmap_workshape_png_u8 = this.#read_dynamic_output_buffer(
                    ringmap_workshape_png_pp,
                    ringmap_workshape_png_size_p
                )
                const cell_info_json_u8:Uint8Array|null = 
                    this.#read_dynamic_output_buffer(
                        cell_info_json_pp,
                        cell_info_json_size_p
                    )
                if(cell_info_json_u8 == null)
                    return new Error('WASM error: did not return cell info')
                const obj:unknown = 
                    JSON.parse(new TextDecoder().decode(cell_info_json_u8));
                cell_info = validate_cell_info_array(obj)
                if(cell_info == null)
                    return new Error('WASM-JS communication inconcistencies')
            }

            if(cellmap_workshape 
            && instancemap_workshape 
            && treeringmap_workshape_shape_png 
            && paired_paths 
            && cell_info
            && ringmap_workshape_png_u8)
                return {
                    cellmap_workshape_png:     cellmap_workshape,
                    cellmap_og_shape_png:      cellmap_og_shape,
                    instancemap_workshape_png: instancemap_workshape,
                    
                    treeringmap_workshape_png: treeringmap_workshape_shape_png,
                    treeringmap_og_shape_png:  treeringmap_og_shape_png,
                    ring_points_xy:            paired_paths,

                    ringmap_workshape_png:
                        new File([ringmap_workshape_png_u8], 'ringmap.png'),
                    cell_info: cell_info,

                    _type: "combined",
                };
            else if(cellmap_workshape && instancemap_workshape)
                return {
                    cellmap_workshape_png:     cellmap_workshape,
                    cellmap_og_shape_png:      cellmap_og_shape,
                    instancemap_workshape_png: instancemap_workshape,
                    
                    _type: "cells"
                }
            else if(treeringmap_workshape_shape_png && paired_paths){
                return {
                    treeringmap_workshape_png: treeringmap_workshape_shape_png,
                    treeringmap_og_shape_png:  treeringmap_og_shape_png,
                    ring_points_xy:            paired_paths,

                    _type: "treerings"
                }
            }
            else
                return new Error('Unexpected error')

        } catch (e) {
            console.error('Unexpected error:', e)
            return e as Error;
        } finally {
            this.#free_allocated_buffers();

            if(cells_handle != 0)
                delete this.#read_file_callback_table[cells_handle];
            if(rings_handle != 0)
                delete this.#read_file_callback_table[rings_handle];

            this.#free_dynamic_buffer_outputs();
        }
    }




    async resize_mask(
        mask:      File, 
        work_size: ImageSize, 
        og_size:   ImageSize
    ): Promise<File|Error> {

        let read_cb_handle:number = 0;

        try {
            read_cb_handle = this.#handle_counter++;
            this.#read_file_callback_table[read_cb_handle] = mask;

            const rc_ptr:number             = this.#malloc(4, /*fill=*/255)
            const resized_png_pp:number     = this.#malloc(8);
            const resized_png_size_p:number = this.#malloc(8);

            let rc:number = this.wasm._resize_mask(
                mask.size, 
                this.#read_file_callback_ptr, 
                read_cb_handle, 

                work_size.width, 
                work_size.height,
                og_size.width, 
                og_size.height,

                resized_png_pp,
                resized_png_size_p,

                rc_ptr
            )
            // NOTE: the wasm function above returns before the execution is 
            // finished because of async issues, so currently polling until done
            while(this.wasm.Asyncify.currData != null)
                await wait(1);
    
            rc = (rc == 0)? this.wasm.HEAP32[rc_ptr >> 2]! : rc;
            if(rc != 0)
                return new Error(`WASM error code = ${rc}`)

            const resized_png_u8:Uint8Array<ArrayBuffer>|null
                = this.#read_dynamic_output_buffer(
                    resized_png_pp,
                    resized_png_size_p
                )
            if(resized_png_u8 == null)
                return new Error(`WASM error: did not return file.`)

            return new File([resized_png_u8], mask.name);

        } catch (e) {
            console.error('Unexpected error:', e)
            return e as Error;
        } finally {
            if(read_cb_handle != 0)
                delete this.#read_file_callback_table[read_cb_handle];
            this.#free_allocated_buffers();
            this.#free_dynamic_buffer_outputs();
        }
    }





    #handle_counter = 1;

    #read_file_callback_ptr:pointer;
    #read_file_callback_table: Record<number, File> = {};

    /** Called by WASM to read a required portion of a file */
    #read_file_callback = (
        handle: number,
        dstbuf: pointer,
        start:  bigint,
        size:   bigint,
    ): unknown  => {
        return this.wasm.Asyncify.handleAsync( async () => {
            const file:File|undefined = 
                this.#read_file_callback_table[handle];
            if(!file)
                return -1;
            
            const slice_u8:Uint8Array = new Uint8Array(
                await file.slice(Number(start), Number(start+size)).arrayBuffer()
            )
            this.wasm.HEAPU8.set(slice_u8, dstbuf);
            return 0;
        })
    }


    #allocated_buffers:pointer[] = []

    #malloc(nbytes:number, fill?:number): pointer {
        const p:pointer = this.wasm._malloc(nbytes);
        this.wasm.HEAPU8.set(new Uint8Array(nbytes).fill(fill ?? 0), p)
        this.#allocated_buffers.push(p);
        return p;
    }

    #free_allocated_buffers() {
        for(const buffer_p of this.#allocated_buffers)
            this.wasm._free(buffer_p);
        this.#allocated_buffers = []
    }



    #dynamic_output_buffers:pointer[] = []

    /** Read an outputbuffer, whose size is not known in advance */
    #read_dynamic_output_buffer(buffer_pp:pointer, size_p:pointer): 
    Uint8Array<ArrayBuffer>|null {
        if(buffer_pp == 0 || size_p == 0)
            return null;

        const buffer_p:pointer = this.wasm.HEAP32[buffer_pp >> 2]!;
        const size:number = Number(this.wasm.HEAP64[size_p >> 3]);
        if(buffer_p == 0 || size == 0)
            return null;

        const data_u8:Uint8Array<ArrayBuffer> = this.wasm.HEAPU8.slice(
            buffer_p, 
            buffer_p + size
        )
        this.#dynamic_output_buffers.push(buffer_pp);
        return data_u8;
    }

    #free_dynamic_buffer_outputs() {
        for(const buffer_pp of this.#dynamic_output_buffers)
            this.wasm._free_output(buffer_pp)
        this.#dynamic_output_buffers = []
    }

}


function validate_point(x:unknown): Point|null {
    if(is_number_array(x) && x.length == 2){
        return x as Point;
    }
    else return null;
}

function validate_path(x:unknown): Path|null {
    if(is_array_of_type(x, validate_point)){
        return x;
    }
    else return null;
}

function validate_pathpair(x:unknown): PathPair|null {
    if(is_array_of_type(x, validate_path) && x.length == 2){
        return x as PathPair;
    }
    else return null;
}

function validate_paired_paths(x:unknown):PairedPaths|null {
    if(is_array_of_type(x, validate_pathpair)){
        return x;
    }
    else return null;
}


function validate_box(x:unknown): [number,number,number,number]|null {
    if(is_number_array(x) && x.length == 4)
            return x as [number,number,number,number];
    else return  null;
}

function validate_cell_info_object(x:unknown): CellInfo|null {
    if(is_object(x)
    && has_number_property(x, "id")
    && has_property_of_type(x, "box_xy", validate_box)
    && has_number_property(x, "year_index")
    && has_number_property(x, "area")
    && has_number_property(x, "position_within")
    ){
        return x;
    }
    else return null;
}

function validate_cell_info_array(x:unknown): CellInfo[]|null {
    if(is_array_of_type(x, validate_cell_info_object)){
        return x;
    }
    else return null;
}




export const initialize:typeof Iinitialize = async () => {
    const wasm:CARROT_Postprocessing_WASM = await (
        await import('./build-wasm/carrot_postprocessing_wasm.js')
    // deno-lint-ignore no-explicit-any
    ).default() as any;

    return new CARROT_Postprocessing(wasm);
}


